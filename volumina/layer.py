import random, colorsys
import numpy

from PyQt4.QtCore import QObject, pyqtSignal, QEvent, Qt
from PyQt4.QtGui import QColor

from widgets.layerDialog import GrayscaleLayerDialog
from widgets.layerDialog import RGBALayerDialog
from volumina.pixelpipeline.datasourcefactories import createDataSource
from volumina.pixelpipeline.asyncabcs import SourceABC

#*******************************************************************************
# L a y e r                                                                    *
#*******************************************************************************

class Layer( QObject ):
    '''
    properties:
    datasources -- list of ArraySourceABC; read-only
    visible -- boolean
    opacity -- float; range 0.0 - 1.0
    name -- string
    layerId -- any object that can uniquely identify this layer within a layerstack (by default, same as name)
    '''

    '''changed is emitted whenever one of the more specialized
    somethingChanged signals is emitted.'''
    changed = pyqtSignal()

    visibleChanged = pyqtSignal(bool) 
    opacityChanged = pyqtSignal(float) 
    nameChanged = pyqtSignal(object)

    @property
    def visible( self ):
        return self._visible
    @visible.setter
    def visible( self, value ):
        if value != self._visible:
            self._visible = value
            self.visibleChanged.emit( value )

    def toggleVisible(self):
        """Convenience function."""
        self.visible = not self._visible

    @property
    def opacity( self ):
        return self._opacity
    @opacity.setter
    def opacity( self, value ):
        if value != self._opacity:
            self._opacity = value
            self.opacityChanged.emit( value )
            
    @property
    def name( self ):
        return self._name
    @name.setter
    def name( self, n ):
        if self._name != n:
            self._name = n
            self.nameChanged.emit(n)

    @property
    def datasources( self ):
        return self._datasources

    @property
    def layerId( self ):
        # If we have no real id, use the name
        if self._layerId is None:
            return self._name
        else:
            return self._layerId
    
    @layerId.setter
    def layerId( self, lid ):
        self._layerId = lid

    def setActive( self, active ):
        """This function is called whenever the layer is selected (active = True) or deselected (active = False)
           by the user.
           As an example, this can be used to enable a specific event interpreter when the layer is active
           and to disable it when it is not active.
           See ClickableColortableLayer for an example."""
        pass

    def timePerTile( self, timeSec, tileRect ):
        """Update the average time per tile with new data: the tile of size tileRect took timeSec seonds"""
        #compute cumulative moving average
        self._numTiles += 1
        self.averageTimePerTile = (timeSec + (self._numTiles-1)*self.averageTimePerTile) / self._numTiles

    def toolTip(self):
        return self._toolTip

    def setToolTip(self, tip):
        self._toolTip = tip

    def __init__( self, direct=False ):
        super(Layer, self).__init__()
        self._name = "Unnamed Layer"
        self._visible = True
        self._opacity = 1.0
        self._datasources = []
        self._layerId = None
        self.direct = direct
        self._toolTip = ""
        
        if self.direct:
            #in direct mode, we calculate the average time per tile for debug purposes
            #this is useful to identify which of your layers cause slowness
            self.averageTimePerTile = 0.0
            self._numTiles = 0

        self.visibleChanged.connect(self.changed)
        self.opacityChanged.connect(self.changed)
        self.nameChanged.connect(self.changed)
        
#*******************************************************************************
# C l i c k a b l e L a y e r                                                  *
#*******************************************************************************

class ClickInterpreter(QObject):
    """Intercepts RIGHT CLICK and double click events on a layer and calls a given functor with the clicked
       position."""
       
    def __init__(self, editor, layer, onClickFunctor, parent=None):
        """ editor:         VolumeEditor object
            layer:          Layer instance on which was clicked
            onClickFunctor: a function f(layer, position5D, windowPosition
        """
        QObject.__init__(self, parent)
        self.baseInterpret = editor.navInterpret
        self.posModel      = editor.posModel
        self._onClick = onClickFunctor
        self._layer = layer

    def start( self ):
        self.baseInterpret.start()

    def stop( self ):
        self.baseInterpret.stop()

    def eventFilter( self, watched, event ):
        ctrl = False
        etype = event.type()
        if etype == QEvent.MouseButtonPress or etype == QEvent.MouseButtonDblClick:
            ctrl = (event.modifiers() == Qt.ControlModifier)
            rightButton = (event.button() == Qt.RightButton)
            leftButton  = (event.button() == Qt.LeftButton)

            if not rightButton:
                return self.baseInterpret.eventFilter(watched, event)
            
            pos = self.posModel.cursorPos
            pos = [int(i) for i in pos]
            pos = [self.posModel.time] + pos + [self.posModel.channel]
            self._onClick(self._layer, tuple(pos), event.pos())
            return True
        
        return self.baseInterpret.eventFilter(watched, event)

class ClickableLayer( Layer ):
    """A layer that, when being activated/selected, switches to an interpreter than can intercept
       right click events"""
    def __init__( self, editor, clickFunctor, direct=False ):
        super(ClickableLayer, self).__init__(direct=direct)
        self._editor = editor
        self._clickInterpreter = ClickInterpreter(editor, self, clickFunctor)
        self._inactiveInterpreter = self._editor.eventSwitch.interpreter
    
    def setActive(self, active):
        if active:
            self._editor.eventSwitch.interpreter = self._clickInterpreter
        else:
            self._editor.eventSwitch.interpreter = self._inactiveInterpreter

#*******************************************************************************
# N o r m a l i z a b l e L a y e r                                            *
#*******************************************************************************

class NormalizableLayer( Layer ):
    '''
    int -- datasource index
    int -- lower threshold
    int -- upper threshold
    '''
    normalizeChanged = pyqtSignal(int, int, int)

    '''
    int -- datasource index
    int -- minimum
    int -- maximum
    '''
    rangeChanged = pyqtSignal(int, int, int)

    @property
    def range( self ):
        return self._range
    def set_range( self, datasourceIdx, value ):
        '''
        value -- (rmin, rmax)
        '''
        self._range[datasourceIdx] = value
        self.rangeChanged.emit(datasourceIdx, value[0], value[1])

    
    @property
    def normalize( self ):
        return self._normalize
    def set_normalize( self, datasourceIdx, value ):
        '''
        value -- (nmin, nmax)
        '''
        self._normalize[datasourceIdx] = value 
        self.normalizeChanged.emit(datasourceIdx, value[0], value[1])

    def __init__( self, direct=False ):
        super(NormalizableLayer, self).__init__(direct=direct)
        self._normalize = []
        self._range = []

        self.rangeChanged.connect(self.changed)
        self.normalizeChanged.connect(self.changed)

#*******************************************************************************
# G r a y s c a l e L a y e r                                                  *
#*******************************************************************************

class GrayscaleLayer( NormalizableLayer ):
    def __init__( self, datasource, range = (0,255), normalize = (0,255), direct=False ):
        assert isinstance(datasource, SourceABC)
        super(GrayscaleLayer, self).__init__(direct=direct)
        self._datasources = [datasource]
        self._normalize = [normalize]
        self._range = [range] 

#*******************************************************************************
# A l p h a M o d u l a t e d L a y e r                                        *
#*******************************************************************************

class AlphaModulatedLayer( NormalizableLayer ):
    tintColorChanged = pyqtSignal()

    @property
    def tintColor(self):
        return self._tintColor
    @tintColor.setter
    def tintColor(self, c):
        if self._tintColor != c:
            self._tintColor = c
            self.tintColorChanged.emit()
    
    def __init__( self, datasource, tintColor = QColor(255,0,0), range = (0,255), normalize = (0,255) ):
        assert isinstance(datasource, SourceABC)
        super(AlphaModulatedLayer, self).__init__()
        self._datasources = [datasource]
        self._normalize = [normalize]
        self._range = [range]
        self._tintColor = tintColor
        self.tintColorChanged.connect(self.changed)
        
#*******************************************************************************
# C o l o r t a b l e L a y e r                                                *
#*******************************************************************************

def generateRandomColors(M=256, colormodel="hsv", clamp=None, zeroIsTransparent=False):
    """Generate a colortable with M entries.
       colormodel: currently only 'hsv' is supported
       clamp:      A dictionary stating which parameters of the color in the colormodel are clamped to a certain
                   value. For example: clamp = {'v': 1.0} will ensure that the value of any generated
                   HSV color is 1.0. All other parameters (h,s in the example) are selected randomly
                   to lie uniformly in the allowed range. """
    r = numpy.random.random((M, 3))
    if clamp is not None:
        for k,v in clamp.iteritems():
            idx = colormodel.index(k)
            r[:,idx] = v

    colors = []
    if colormodel == "hsv":
        for i in range(M):
            if zeroIsTransparent and i == 0:
                colors.append(QColor(0, 0, 0, 0).rgba())
            else:
                h, s, v = r[i,:] 
                color = numpy.asarray(colorsys.hsv_to_rgb(h, s, v)) * 255
                qColor = QColor(*color)
                colors.append(qColor.rgba())
        return colors
    else:
        raise RuntimeError("unknown color model '%s'" % colormodel)

class ColortableLayer( Layer ):
    colorTableChanged = pyqtSignal()

    @property
    def colorTable( self ):
        return self._colorTable

    @colorTable.setter
    def colorTable( self, colorTable ):
        self._colorTable = colorTable
        self.colorTableChanged.emit()

    def randomizeColors(self):
        self.colorTable = generateRandomColors(len(self._colorTable), "hsv", {"v": 1.0}, True)

    def __init__( self, datasource , colorTable, direct=False ):
        assert isinstance(datasource, SourceABC)
        super(ColortableLayer, self).__init__(direct=direct)
        self._datasources = [datasource]
        self.data = datasource
        self._colorTable = colorTable
        
        self.colortableIsRandom = False
        self.zeroIsTransparent  = False
        
class ClickableColortableLayer(ClickableLayer):
    colorTableChanged = pyqtSignal()
    
    def __init__( self, editor, clickFunctor, datasource , colorTable, direct=False ):
        assert isinstance(datasource, SourceABC)
        super(ClickableColortableLayer, self).__init__(editor, clickFunctor, direct=direct)
        self._datasources = [datasource]
        self._colorTable = colorTable
        self.data = datasource
        
        self.colortableIsRandom = False
        self.zeroIsTransparent  = False

    @property
    def colorTable( self ):
        return self._colorTable

    @colorTable.setter
    def colorTable( self, colorTable ):
        self._colorTable = colorTable
        self.colorTableChanged.emit()

    def randomizeColors(self):
        self.colorTable = generateRandomColors(len(self._colorTable), "hsv", {"v": 1.0}, True)

#*******************************************************************************
# R G B A L a y e r                                                            *
#*******************************************************************************

class RGBALayer( NormalizableLayer ):
    channelIdx = {'red': 0, 'green': 1, 'blue': 2, 'alpha': 3}
    channelName = {0: 'red', 1: 'green', 2: 'blue', 3: 'alpha'}
    
    @property
    def color_missing_value( self ):
        return self._color_missing_value

    @property
    def alpha_missing_value( self ):
        return self._alpha_missing_value

    def __init__( self, red = None, green = None, blue = None, alpha = None, \
                  color_missing_value = 0, alpha_missing_value = 255,
                  range = 4*[(0,255)],
                  normalizeR=(0,255), normalizeG=(0,255), normalizeB=(0,255), normalizeA=(0,255)):
        assert red is None or isinstance(red, SourceABC)
        assert green is None or isinstance(green, SourceABC)
        assert blue is None or isinstance(blue, SourceABC)
        assert alpha is None or isinstance(alpha, SourceABC)
        super(RGBALayer, self).__init__()
        self._datasources = [red,green,blue,alpha]
        self._normalize   = [normalizeR, normalizeG, normalizeB, normalizeA]
        self._color_missing_value = color_missing_value
        self._alpha_missing_value = alpha_missing_value
        self._range = range

    @classmethod
    def createFromMultichannel(cls, data):
        # disect data
        l = RGBALayer()
        return l
