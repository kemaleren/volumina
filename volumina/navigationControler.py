from PyQt4.QtCore import QObject, QTimer, QEvent, Qt, QPointF, pyqtSignal, \
                         QRectF, QPoint
from PyQt4.QtGui  import QColor, QCursor, QMouseEvent, QApplication, \
                         QPainter, QPen, QGraphicsView, QGraphicsTextItem

import copy
from functools import partial

from imageView2D import ImageView2D
from imageScene2D import ImageScene2D
from eventswitch import InterpreterABC

def posView2D(pos3d, axis):
    """convert from a 3D position to a 2D position on the slicing plane
       perpendicular to axis"""
    pos2d = copy.deepcopy(pos3d)
    del pos2d[axis]
    return pos2d

#*******************************************************************************
# N a v i g a t i o n I n t e r p r e t e r                                    *
#*******************************************************************************

class NavigationInterpreter(QObject):
    # states
    FINAL = 0
    DEFAULT_MODE = 1
    DRAG_MODE = 2

    @property
    def state( self ):
        return self._current_state

    def __init__(self, navigationcontroler):
        QObject.__init__(self)
        self._navCtrl = navigationcontroler
        self._current_state = self.FINAL

    def start( self ):
        if self._current_state == self.FINAL:
            self._current_state = self.DEFAULT_MODE
        else:
            pass # ignore

    def stop( self ):
        self._current_state = self.FINAL

    def eventFilter( self, watched, event ):
        if not self._navCtrl.enableNavigation:
            return False

        etype = event.type()
        ### the following implements a simple state machine
        if self._current_state == self.DEFAULT_MODE:
            ### default mode -> drag mode
            if    (etype == QEvent.MouseButtonPress and event.button() == Qt.MidButton) \
               or (etype == QEvent.MouseButtonPress and event.modifiers() == Qt.ShiftModifier):
                # self.onExit_default(): call it here, if needed
                self._current_state = self.DRAG_MODE
                self.onEntry_drag( watched, event )
                event.accept()
                return True

            ### actions in default mode
            elif etype == QEvent.MouseMove:
                return self.onMouseMove_default( watched, event )

            elif etype == QEvent.Wheel:
                 self.onWheel_default( watched, event )
                 event.accept()
                 return True

            elif etype == QEvent.MouseButtonDblClick:
                self.onMouseDoubleClick_default( watched, event )
                event.accept()
                return True

            elif etype == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self.onMousePressRight_default( watched, event )
                event.accept()
                return True

        elif self._current_state == self.DRAG_MODE:
            ### drag mode -> default mode
            if etype == QEvent.MouseButtonRelease:
                self.onExit_drag( watched, event)
                self._current_state = self.DEFAULT_MODE
                self.onEntry_default( watched, event )
                event.accept()
                return True

            ### actions in drag mode
            elif etype == QEvent.MouseMove:
                self.onMouseMove_drag( watched, event )
                event.accept()
                return True

        event.ignore()
        return False

    ###
    ### Default Mode
    ###
    def onEntry_default( self, imageview, event ):
        pass

    def onWheel_default( self, imageview, event ):
        navCtrl = self._navCtrl
        k_alt = (event.modifiers() == Qt.AltModifier)
        k_ctrl = (event.modifiers() == Qt.ControlModifier)
        k_shift = (event.modifiers() == Qt.ShiftModifier)
        k_shift_alt = (event.modifiers() == (Qt.ShiftModifier | Qt.AltModifier))

        imageview.mousePos = imageview.mapScene2Data(imageview.mapToScene(event.pos()))

        sceneMousePos = imageview.mapToScene(event.pos())
        grviewCenter = imageview.mapToScene(imageview.viewport().rect().center())

        if event.delta() < 0:
            if k_shift_alt:
                navCtrl.changeTimeRelative(-10)
            elif k_alt:
                navCtrl.changeSliceRelative(-10, navCtrl._views.index(imageview))
            elif k_ctrl:
                scaleFactor = 0.9
                imageview.doScale(scaleFactor)
            elif k_shift:
                navCtrl.changeTimeRelative(-1)
            else:
                navCtrl.changeSliceRelative(-1, navCtrl._views.index(imageview))
        else:
            if k_shift_alt:
                navCtrl.changeTimeRelative(10)
            elif k_alt:
                navCtrl.changeSliceRelative(10, navCtrl._views.index(imageview))
            elif k_ctrl:
                scaleFactor = 1.1
                imageview.doScale(scaleFactor)
            elif k_shift:
                navCtrl.changeTimeRelative(1)
            else:
                navCtrl.changeSliceRelative(1, navCtrl._views.index(imageview))
        if k_ctrl:
            mousePosAfterScale = imageview.mapToScene(event.pos())
            offset = sceneMousePos - mousePosAfterScale
            newGrviewCenter = grviewCenter + offset
            imageview.centerOn(newGrviewCenter)
            self.onMouseMove_default( imageview, event )

    def onMouseMove_default( self, imageview, event ):
        if imageview._ticker.isActive():
            #the view is still scrolling
            #do nothing until it comes to a complete stop
            return False

        imageview.mousePos = mousePos = imageview.mapMouseCoordinates2Data(event.pos())
        imageview.oldX, imageview.oldY = imageview.x, imageview.y
        dataX = imageview.x = mousePos.x()
        dataY = imageview.y = mousePos.y()

        return self._navCtrl.positionDataCursor(QPointF(dataX, dataY), self._navCtrl._views.index(imageview))

    def onMousePressRight_default( self, imageview, event ):
        #make sure that we have the cursor at the correct position
        #before we call the context menu
        self.onMouseMove_default( imageview, event )
        pos = event.pos()
        imageview.customContextMenuRequested.emit( pos )

    def onMouseDoubleClick_default( self, imageview, event ):
        dataMousePos = imageview.mapScene2Data(imageview.mapToScene(event.pos()))
        self._navCtrl.navigateToPoint(dataMousePos.x(), dataMousePos.y(), self._navCtrl._views.index(imageview))

    ###
    ### Drag Mode
    ###
    def onEntry_drag( self, imageview, event ):
        imageview.setCursor(QCursor(Qt.SizeAllCursor))
        imageview._lastPanPoint = event.pos()
        imageview._crossHairCursor.setVisible(False)
        imageview._dragMode = True
        if imageview._ticker.isActive():
            imageview._deltaPan = QPointF(0, 0)

    def onExit_drag( self, imageview, event ):
        imageview.mousePos = imageview.mapScene2Data(imageview.mapToScene(event.pos()))
        imageview.setCursor(QCursor())
        releasePoint = event.pos()
        imageview._lastPanPoint = releasePoint
        imageview._dragMode = False
        imageview._ticker.start(20)

    def onMouseMove_drag( self, imageview, event ):
        imageview._deltaPan = QPointF(event.pos() - imageview._lastPanPoint)
        imageview._panning()
        imageview._lastPanPoint = event.pos()
assert issubclass(NavigationInterpreter, InterpreterABC)

#*******************************************************************************
# N a v i g a t i o n C o n t r o l e r                                        *
#*******************************************************************************

class NavigationControler(QObject):
    """
    Controler for navigating through the volume.

    The NavigationContrler object listens to changes
    in a given PositionModel and updates three slice
    views (representing the spatial X, Y and Z slicings)
    accordingly.

    properties:

    indicateSliceIntersection -- whether to show red/green/blue lines
        indicating the position of the other two slice views on each slice
        view

    enableNavigation -- whether the position is allowed to be changed
    """

    navigationEnabled = pyqtSignal(bool)

    @property
    def axisColors( self ):
        return self._axisColors
    @axisColors.setter
    def axisColors( self, colors ):
        self._axisColors = colors
        self._views[0]._sliceIntersectionMarker.setColor(self.axisColors[2], self.axisColors[1])
        self._views[1]._sliceIntersectionMarker.setColor(self.axisColors[2], self.axisColors[0])
        self._views[2]._sliceIntersectionMarker.setColor(self.axisColors[1], self.axisColors[0])
        for axis, v in enumerate(self._views):
            #FIXME: Bad dependency here on hud to be available!
            if v.hud: v.hud.bgColor = self.axisColors[axis]

    @property
    def indicateSliceIntersection(self):
        return self._indicateSliceIntersection
    @indicateSliceIntersection.setter
    def indicateSliceIntersection(self, show):
        self._indicateSliceIntersection = show
        for v in self._views:
            v._sliceIntersectionMarker.setVisibility(show)

    @property
    def enableNavigation(self):
        return self._navigationEnabled
    @enableNavigation.setter
    def enableNavigation(self, enabled):
        self._navigationEnabled = enabled
        self.navigationEnabled.emit(enabled)

    def __init__(self, imageView2Ds, sliceSources, positionModel, time = 0, channel = 0, view3d=None):
        QObject.__init__(self)
        assert len(imageView2Ds) == 3

        # init fields
        self._views = imageView2Ds
        self._sliceSources = sliceSources
        self._model = positionModel
        self._beginStackIndex = 0
        self._endStackIndex   = 1
        self._view3d = view3d
        self._navigationEnabled = True

        self.axisColors = [QColor(255,0,0,255), QColor(0,255,0,255), QColor(0,0,255,255)]

    def moveCrosshair(self, newPos, oldPos):
        self._updateCrossHairCursor()

    def navigateToPoint(self, x, y, axis):
        newPos = copy.copy(self._model.slicingPos)
        i,j = posView2D([0,1,2], axis)
        newPos[i] = x
        newPos[j] = y
        if newPos == self._model.slicingPos:
            return
        if not self._positionValid(newPos):
            return

        self._model.slicingPos = newPos
        self.panSlicingViews( newPos, filter( lambda a: a != axis, [0,1,2] ) )

    def panSlicingViews(self, point3d, axes):
        """
        For each of the given axes, pan the slicing view to the ordinate-abscissa point in point3d,
        but DON'T change the slicing plane.
        """
        for axis, view in enumerate(self._views):
            if axis in axes:
                pos2d = posView2D(point3d, axis)
                dataPoint = QPointF( *pos2d )
                scenePoint = view.scene().data2scene.map(dataPoint)
                view.centerOn( scenePoint )

    def moveSlicingPosition(self, newPos, oldPos):
        for i in range(3):
            if newPos[i] != oldPos[i]:
                self._updateSlice(self._model.slicingPos[i], i)
        self._updateSliceIntersection()

        #when scrolling fast through the stack, we don't want to update
        #the 3d view all the time
        if self._view3d is None:
            return
        def maybeUpdateSlice(oldSlicing):
            if oldSlicing == self._model.slicingPos:
                for i in range(3):
                    self._view3d.ChangeSlice(self._model.slicingPos[i], i)
        QTimer.singleShot(50, partial(maybeUpdateSlice, self._model.slicingPos))

    def changeTime(self, newTime):
        for i in range(3):
            self._sliceSources[i].setThrough(0, newTime)

    def changeTimeRelative( self, delta ):
        if self._model.shape5D is None or delta == 0:
            return
        cur_t = self._sliceSources[0].through[0]
        new_t = cur_t + delta

        #sanitize
        new_t = 0 if new_t < 0 else new_t
        new_t = self._model.shape5D[0] - 1 if new_t >= self._model.shape5D[0] else new_t
        self._model.time = new_t

    def changeChannel(self, newChannel):
        if self._model.shape is None:
            return
        for i in range(3):
            self._sliceSources[i].setThrough(2, newChannel)

    def changeSliceRelative(self, delta, axis):
        if self._model.shape is None:
            return
        """
        Change slice along a certain axis relative to current slice.

        delta -- add delta to current slice position [positive or negative int]
        axis  -- along which axis [0,1,2]
        """

        if delta == 0:
            return
        newSlice = self._model.slicingPos[axis] + delta
        if newSlice < 0 or newSlice >= self._model.volumeExtent(axis):
            return
        newPos = copy.copy(self._model.slicingPos)
        newPos[axis] = newSlice

        cursorPos = copy.copy(self._model.cursorPos)
        cursorPos[axis] = newSlice
        self._model.cursorPos  = cursorPos

        self._model.slicingPos = newPos

    def changeSliceAbsolute(self, value, axis):
        """
        Change slice along a certain axis.

        value -- slice number
        axis  -- along which axis [0,1,2]
        """

        if value < 0 or value > self._model.volumeExtent(axis):
            return
        newPos = copy.copy(self._model.slicingPos)
        newPos[axis] = value
        if not self._positionValid(newPos):
            return

        cursorPos = copy.copy(self._model.cursorPos)
        cursorPos[axis] = value
        self._model.cursorPos  = cursorPos

        self._model.slicingPos = newPos

    def settleSlicingPosition(self, settled):
        for v in self._views:
            v.indicateSlicingPositionSettled(settled)


    def positionDataCursor(self, dataCoord2D, axis):
        """
        Change position of the crosshair cursor.
        dataCord2D -- 2D coordinate on the slicing plane perpendicular to axis
                      in data coordinate system
        axis -- perpendicular axis [0,1,2]
        """

        #we get the 2D coordinates x,y from the view that
        #shows the projection perpendicular to axis
        #set this view as active
        self._model.activeView = axis

        newPos = [dataCoord2D.x(), dataCoord2D.y()]
        newPos.insert(axis, self._model.slicingPos[axis])

        if not self._positionValid(newPos):
            return False

        if newPos == self._model.cursorPos:
            return True

        self._model.cursorPos = newPos

        return True

    #def positionCursor(self, scenePoint, axis):
    #    """
    #    Change position of the crosshair cursor.

    #    axis -- perpendicular axis [0,1,2]
    #    """
    #
    #    #we get the 2D coordinates x,y from the view that
    #    #shows the projection perpendicular to axis
    #    #set this view as active
    #    self._model.activeView = axis
    #
    #    dataPoint = self._views[axis].scene().scene2data.map(scenePoint)

    #    newPos = [dataPoint.x(), dataPoint.y()]
    #    newPos.insert(axis, self._model.slicingPos[axis])

    #    if newPos == self._model.cursorPos:
    #        return
    #    if not self._positionValid(newPos):
    #        return

    #    self._model.cursorPos = newPos

    #private functions ########################################################

    def _updateCrossHairCursor(self):
        dataX, dataY = posView2D(self._model.cursorPos, axis=self._model.activeView)

        self._views[self._model.activeView]._crossHairCursor.showXYPosition(dataX, dataY)
        for i, v in enumerate(self._views):
            v._crossHairCursor.setVisible( self._model.activeView == i )

    def _updateSliceIntersection(self):
        for axis, v in enumerate(self._views):
            dataX, dataY = posView2D(self._model.slicingPos, axis)
            v._sliceIntersectionMarker.setPosition( dataX, dataY )

    def _updateSlice(self, num, axis):
        if num < 0 or num >= self._model.volumeExtent(axis):
            raise Exception("NavigationControler._setSlice(): invalid slice number = %d not in range [0,%d)" % (num, self._model.volumeExtent(axis)))
        #FIXME: Shouldnt the hud listen to the model changes itself?
        self._views[axis].hud.sliceSelector.setValue(num)

        #re-configure the slice source
        self._sliceSources[axis].setThrough(1,num)

    def _positionValid(self, pos):
        if self._model.shape is None:
            return False
        for i in range(3):
            if pos[i] < 0 or pos[i] >= self._model.shape[i]:
                return False
        return True
