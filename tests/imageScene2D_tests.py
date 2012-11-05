import unittest as ut
import os

from PyQt4.QtGui import QImage, QPainter, QApplication

from qimage2ndarray import byte_view
import numpy as np

from volumina.imageScene2D import ImageScene2D
from volumina.positionModel import PositionModel
from volumina.pixelpipeline.datasources import ConstantSource
from volumina.pixelpipeline.imagepump import StackedImageSources
from volumina.layerstack import LayerStackModel
from volumina.layer import GrayscaleLayer
import volumina.pixelpipeline.imagesourcefactories as imsfac

class ImageScene2DTest( ut.TestCase ):

    @classmethod
    def setUpClass(cls):
        cls.app = None
        if QApplication.instance():
            cls.app = QApplication.instance()
        else:
            cls.app = QApplication([], False)

    @classmethod
    def tearDownClass(cls):
        del cls.app

    def testStackedImageSourcesProperty( self ):
        s = ImageScene2D(PositionModel(), (0,3,4), preemptive_fetch_number=0)
        self.assertEqual(len(s.stackedImageSources), 0)

        sims = StackedImageSources( LayerStackModel() )
        s.stackedImageSources = sims
        self.assertEqual(id(s.stackedImageSources), id(sims))

class ImageScene2D_RenderTest( ut.TestCase ):

    @classmethod
    def setUpClass(cls):
        cls.app = None
        if QApplication.instance():
            cls.app = QApplication.instance()
        else:
            cls.app = QApplication([], False)

    @classmethod
    def tearDownClass(cls):
        del cls.app

    def setUp( self ):
        self.layerstack = LayerStackModel()
        self.sims = StackedImageSources( self.layerstack )

        self.GRAY = 201
        self.ds = ConstantSource(self.GRAY)
        self.layer = GrayscaleLayer( self.ds )
        self.layerstack.append(self.layer)
        self.ims = imsfac.createImageSource( self.layer, [self.ds] )
        self.sims.register(self.layer, self.ims)

        self.scene = ImageScene2D(PositionModel(), (0,3,4), preemptive_fetch_number=0)

        self.scene.stackedImageSources = self.sims
        self.scene.dataShape = (310,290)

    def tearDown( self ):
        if self.scene._tileProvider:
            self.scene._tileProvider.notifyThreadsToStop()
            self.scene._tileProvider.joinThreads()

    def renderScene( self, s):
        img = QImage(310,290,QImage.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        s.render(p)
        s.joinRendering()
        s.render(p)
        p.end()
        return byte_view(img)

    def testBasicImageRenderingCapability( self ):
        aimg = self.renderScene(self.scene)
        self.assertTrue(np.all(aimg[:,:,0:3] == self.GRAY))
        self.assertTrue(np.all(aimg[:,:,3] == 255))

    @ut.skipIf(os.getenv('TRAVIS'), 'fails on TRAVIS CI due to unknown reasons')
    def testToggleVisibilityOfOneLayer( self ):
        aimg = self.renderScene(self.scene)
        self.assertTrue(np.all(aimg[:,:,0:3] == self.GRAY))
        self.assertTrue(np.all(aimg[:,:,3] == 255))

        self.layer.visible = False
        aimg = self.renderScene(self.scene)
        self.assertTrue(np.all(aimg[:,:,0:3] == 255)) # all white
        self.assertTrue(np.all(aimg[:,:,3] == 255))

        self.layer.visible = True
        aimg = self.renderScene(self.scene)
        self.assertTrue(np.all(aimg[:,:,0:3] == self.GRAY))
        self.assertTrue(np.all(aimg[:,:,3] == 255))

if __name__ == '__main__':
    ut.main()
