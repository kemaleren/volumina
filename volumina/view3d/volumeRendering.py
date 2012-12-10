import vtk
import h5py
import numpy
import colorsys
# http://www.scipy.org/Cookbook/vtkVolumeRendering

def makeVolumeRenderingPipeline(in_volume):
    dataImporter = vtk.vtkImageImport()

    if in_volume.dtype == numpy.uint8:
        dataImporter.SetDataScalarTypeToUnsignedChar()
    elif in_volume.dtype == numpy.uint16:
        dataImporter.SetDataScalarTypeToUnsignedShort()
    elif in_volume.dtype == numpy.int32:
        dataImporter.SetDataScalarTypeToInt()
    elif in_volume.dtype == numpy.int16:
        dataImporter.SetDataScalarTypeToShort()
    else:
        raise RuntimeError("unknown data type %r of volume" % (in_volume.dtype,))

    dataImporter.SetImportVoidPointer(in_volume, len(in_volume))
    dataImporter.SetNumberOfScalarComponents(1)
    extent = [0, in_volume.shape[0]-1, 0, in_volume.shape[1]-1, 0, in_volume.shape[2]-1]
    dataImporter.SetDataExtent(*extent)
    dataImporter.SetWholeExtent(*extent)

    # The following class is used to store transparency-values for
    # later retrieval. In our case, we want the value 0 to be
    # completely opaque whereas the three different cubes are given
    # different transparency-values to show how it works.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    for i in range(1, 256):
        alphaChannelFunc.AddPoint(i, 1.0)

    # This class stores color data and can create color tables from a
    # few color points. For this demo, we want the three cubes to be
    # of the colors red green and blue.

    colorFunc = vtk.vtkColorTransferFunction()
    '''
    for i in range(1, maxLabel+1):
        rgb = colorsys.hsv_to_rgb(numpy.random.random(), 1.0, 1.0)
        colorFunc.AddRGBPoint(i, *rgb)
    '''

    # The previous two classes stored properties. Because we want to
    # apply these properties to the volume we want to render, we have
    # to store them in a class that stores volume properties.

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)

    smart = True
    if smart:
        volumeMapper = vtk.vtkSmartVolumeMapper()
        #volumeMapper.SetRequestedRenderMode(vtk.vtkSmartVolumeMapper.GPURenderMode)
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
        volumeProperty.ShadeOff()

        #volumeProperty.ShadeOn()
        #volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)
        volumeProperty.SetInterpolationType(vtk.VTK_NEAREST_INTERPOLATION)
    else:
        #volumeProperty.ShadeOn()

        # This class describes how the volume is rendered (through ray tracing).
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

        # We can finally create our volume. We also have to specify
        # the data for it, as well as how the data will be rendered.
        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())


    # The class vtkVolume is used to pair the previously declared
    # volume as well as the properties to be used when rendering that
    # volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return dataImporter, colorFunc, volume


class LabelManager(object):
    def __init__(self, n):
        self._available = set(range(n))
        self._used = set([])

    def request(self):
        if len(self._available) == 0:
            raise RuntimeError('out of labels')
        label = min(self._available)
        self._available.remove(label)
        self._used.add(label)
        return label

    def free(self, label):
        if label in self._used:
            self._used.remove(label)
            self._available.add(label)


class RenderingManager(object):
    """Encapsulates the work of adding/removing objects to the
    rendered volume and setting their colors.

    Conceptually very simple: given a volume containing integer labels
    (where zero labels represent transparent background) and a color
    map, renders the objects in the appropriate color.

    """
    def __init__(self, shape, renderer, qvtk=None):
        self._renderer = renderer
        self._qvtk = qvtk
        self.labelmgr = LabelManager(256)
        self._volume = numpy.zeros(shape, dtype=numpy.uint8)
        self._ready = False
        self._initialize()

    def _initialize(self):
        dataImporter, colorFunc, volume = makeVolumeRenderingPipeline(self._volume)
        self._renderer.AddVolume(volume)
        self._volumeRendering = volume
        self._dataImporter = dataImporter
        self._colorFunc = colorFunc
        self._ready = True

    def update(self):
        """Only needs to be called directly if new data has manually
        been written to the volume.

        """
        self._dataImporter.Modified()
        self._volumeRendering.Update()
        if self._qvtk is not None:
            self._qvtk.update()

    def addObject(self, indices, color=None):
        label = self.labelmgr.request()
        self._volume[indices] = label
        if color is None:
            color = colorsys.hsv_to_rgb(numpy.random.random(), 1.0, 1.0)
        self._colorFunc.AddRGBPoint(label, *color)
        self.update()
        return label

    def removeObject(self, label):
        self._volume[numpy.where(self._volume == label)] = 0
        self.labelmgr.free(label)
        self.update()

    def updateColorMap(self, cmap):
        for label, color in cmap.iteritems():
            self._colorFunc.AddRGBPoint(label, *color)
        self.update()

    def clear(self, ):
        self._volume[:] = 0


if __name__ == "__main__":

    # With almost everything else ready, its time to initialize the
    # renderer and window, as well as creating a method for exiting
    # the application
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)
    renderer.SetBackground(1, 1, 1) # white background
    renderWin.SetSize(400, 400)

    # A simple function to be called when the user decides to quit the
    # application.
    def exitCheck(obj, event):
        if obj.GetEventPending() != 0:
            obj.SetAbortRender(1)

    # Tell the application to use the function as an exit check.
    renderWin.AddObserver("AbortCheckEvent", exitCheck)

    # create the rendering manager
    mgr = RenderingManager((256, 256, 256), renderer)
    mgr.addObject([slice(50, 150)] * 3)

    renderInteractor.Initialize()

    # Because nothing will be rendered without any input, we order the
    # first render manually before control is handed over to the
    # main-loop.
    renderWin.Render()

    renderInteractor.Start()