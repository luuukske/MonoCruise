"""Popup scale/fade animations for PySide6 widgets."""

from typing import Optional
from PySide6.QtCore import (
    QObject, QPropertyAnimation, QParallelAnimationGroup,
    QEasingCurve, Signal, QPoint, Property, QRectF, Qt
)
from PySide6.QtWidgets import (
    QWidget, QGraphicsOpacityEffect, QGraphicsView, 
    QGraphicsScene, QGraphicsProxyWidget, QApplication
)
from PySide6.QtGui import QTransform, QPainter, QColor


class ScalableContainer(QGraphicsView):
    """Graphics view: base_scale (DPI) plus anim_scale for popup animations."""
    
    def __init__(self, content_widget: QWidget, parent: QWidget = None):
        super().__init__(parent)
        
        self.setObjectName("ScalableContainer")
        
        self._content_size = content_widget.size()
        self._base_scale = 1.0
        self._anim_scale = 1.0
        
        # Set up the scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        # Set scene rect to match content
        self._scene.setSceneRect(QRectF(0, 0, self._content_size.width(), self._content_size.height()))
        
        # Add the content widget as a proxy
        self._proxy = QGraphicsProxyWidget()
        self._proxy.setWidget(content_widget)
        self._scene.addItem(self._proxy)
        
        # Configure the view for seamless appearance - NO BACKGROUND
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
        # Make fully transparent - both view and viewport
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent; border: none;")
        self.viewport().setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.viewport().setStyleSheet("background: transparent;")
        self.setAutoFillBackground(False)
        self.viewport().setAutoFillBackground(False)
        
        # Also set scene background to transparent
        self._scene.setBackgroundBrush(Qt.GlobalColor.transparent)
        
        # Match size to content
        self.setFixedSize(self._content_size)
    
    def _apply_transform(self):
        """Rebuild the combined transform from base_scale and anim_scale."""
        self.resetTransform()
        combined = self._base_scale * self._anim_scale
        
        if self._anim_scale == 1.0:
            # Pure base scale: scale from top-left origin
            transform = QTransform()
            transform.scale(combined, combined)
        else:
            # Animation in progress: center the anim_scale on the
            # *scaled* viewport so the shrink/grow looks centred.
            vw = self._content_size.width() * self._base_scale
            vh = self._content_size.height() * self._base_scale
            cx = vw / 2
            cy = vh / 2
            transform = QTransform()
            transform.translate(cx, cy)
            transform.scale(combined, combined)
            transform.translate(-cx / self._base_scale, -cy / self._base_scale)
        
        self.setTransform(transform)
    
    def set_base_scale(self, scale: float):
        """Set the permanent resolution scale. Resizes the viewport accordingly."""
        self._base_scale = scale
        w = int(self._content_size.width() * scale)
        h = int(self._content_size.height() * scale)
        self.setFixedSize(w, h)
        self._apply_transform()
    
    def set_anim_scale(self, scale: float):
        """Set the animation scale (centered shrink/grow)."""
        self._anim_scale = scale
        self._apply_transform()
    
    def get_anim_scale(self) -> float:
        return self._anim_scale
    
    def get_proxy(self) -> QGraphicsProxyWidget:
        return self._proxy


class PopupAnimator(QObject):
    animation_finished = Signal(str)
    
    DURATION_SLIDE = 400
    DURATION_SCALE = 250
    
    SCALE_FULL = 1.0
    SCALE_SMALL = 0.8
    
    def __init__(
        self,
        widget: QWidget,
        content_widget: QWidget,
        opacity_effect: QGraphicsOpacityEffect,
        target_y: int,
        hidden_y: int
    ):
        super().__init__()
        
        self._widget = widget
        self._opacity = opacity_effect
        self._target_y = target_y
        self._hidden_y = hidden_y
        
        self._animation: Optional[QParallelAnimationGroup] = None
        self._current_scale = self.SCALE_FULL
        
        # Wrap content in scalable container
        self._scalable = ScalableContainer(content_widget, widget)
        self._scalable.move(0, 0)
        self._scalable.show()
    
    def _get_x(self) -> int:
        return self._widget.pos().x()
    
    def _stop_animations(self):
        if self._animation:
            self._animation.stop()
            try:
                self._animation.finished.disconnect()
            except TypeError:
                pass
            self._animation = None
    
    def _create_pos_animation(self, start_y: int, end_y: int, duration: int) -> QPropertyAnimation:
        anim = QPropertyAnimation(self._widget, b"pos")
        anim.setDuration(duration)
        anim.setStartValue(QPoint(self._get_x(), start_y))
        anim.setEndValue(QPoint(self._get_x(), end_y))
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        return anim
    
    def _create_opacity_animation(self, start: float, end: float, duration: int) -> QPropertyAnimation:
        anim = QPropertyAnimation(self._opacity, b"opacity")
        anim.setDuration(duration)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        return anim

    # CUSTOM SCALE PROPERTY FOR ANIMATION
    
    def _get_scale(self) -> float:
        return self._current_scale
    
    def _apply_scale(self, scale: float):
        self._current_scale = scale
        self._scalable.set_anim_scale(scale)
    
    # Define as a Property so QPropertyAnimation can use it
    scale = Property(float, _get_scale, _apply_scale)
    
    def set_base_scale(self, scale: float):
        """Set the permanent resolution scale on the scalable container."""
        self._scalable.set_base_scale(scale)

    def _create_scale_animation(self, start: float, end: float, duration: int) -> QPropertyAnimation:
        anim = QPropertyAnimation(self, b"scale")
        anim.setDuration(duration)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        return anim
    
    def _run_animation(self, animation_type: str):
        self._animation.finished.connect(lambda: self.animation_finished.emit(animation_type))
        self._animation.start()
    
    def _move_to_visible(self):
        """Move widget to visible position without animation."""
        self._widget.move(self._get_x(), self._target_y)
    
    # Public Animation Methods
    
    def slide_in(self):
        self._stop_animations()
        self._apply_scale(self.SCALE_FULL)
        
        self._animation = QParallelAnimationGroup(self)
        self._animation.addAnimation(
            self._create_pos_animation(self._hidden_y, self._target_y, self.DURATION_SLIDE)
        )
        self._animation.addAnimation(
            self._create_opacity_animation(0.0, 1.0, self.DURATION_SLIDE)
        )
        self._run_animation("slide_in")
    
    def slide_out(self):
        self._stop_animations()
        
        self._animation = QParallelAnimationGroup(self)
        self._animation.addAnimation(
            self._create_pos_animation(self._target_y, self._hidden_y, self.DURATION_SLIDE)
        )
        self._animation.addAnimation(
            self._create_opacity_animation(1.0, 0.0, self.DURATION_SLIDE)
        )
        self._run_animation("slide_out")
    
    def scale_out(self):
        self._stop_animations()
        
        self._animation = QParallelAnimationGroup(self)
        self._animation.addAnimation(
            self._create_opacity_animation(1.0, 0.0, self.DURATION_SCALE)
        )
        self._animation.addAnimation(
            self._create_scale_animation(self.SCALE_FULL, self.SCALE_SMALL, self.DURATION_SCALE)
        )
        self._run_animation("scale_out")

    def scale_in(self):
        self._stop_animations()
        self._apply_scale(self.SCALE_SMALL)  # Start small
        self._move_to_visible()
        
        self._animation = QParallelAnimationGroup(self)
        self._animation.addAnimation(
            self._create_opacity_animation(0.0, 1.0, self.DURATION_SCALE)
        )
        self._animation.addAnimation(
            self._create_scale_animation(self.SCALE_SMALL, self.SCALE_FULL, self.DURATION_SCALE)
        )
        self._run_animation("scale_in")
    
    def update_geometry(self, target_y: int, hidden_y: int):
        """Update positions after a scale change."""
        self._target_y = target_y
        self._hidden_y = hidden_y
    
    def reset(self):
        self._stop_animations()
        self._widget.move(self._get_x(), self._hidden_y)
        self._opacity.setOpacity(0.0)
        self._apply_scale(self.SCALE_FULL)
