"""
Body Shape Classification Module
Classifies body shapes based on fashion industry standards
"""

from typing import Dict, Any, Optional


class BodyShapeClassifier:
    """
    Classifies body shapes using shoulder-to-hip ratios and waist measurements
    according to fashion industry standards.
    """
    
    # Classification thresholds (fashion industry standards)
    RATIO_THRESHOLD_HIGH = 1.20  # Shoulders significantly wider than hips
    RATIO_THRESHOLD_LOW = 0.95   # Hips significantly wider than shoulders
    WAIST_DEFINITION_THRESHOLD = 0.20  # 20% difference for rectangle classification
    HOURGLASS_WAIST_RATIO = 0.80  # Waist must be < 80% of hips for hourglass
    
    def __init__(self) -> None:
        """Initialize the body shape classifier."""
        pass
    
    def _calculate_shoulder_hip_ratio(
        self, 
        shoulder_width: float, 
        hip_width: float
    ) -> float:
        """
        Calculate the shoulder-to-hip width ratio.
        
        Args:
            shoulder_width: Shoulder width in pixels
            hip_width: Hip width in pixels
            
        Returns:
            Ratio of shoulder width to hip width
        """
        if hip_width == 0:
            return 0.0
        
        return shoulder_width / hip_width
    
    def _calculate_waist_definition(
        self,
        waist_width: float,
        shoulder_width: float,
        hip_width: float
    ) -> float:
        """
        Calculate waist definition percentage.
        
        Args:
            waist_width: Waist width in pixels
            shoulder_width: Shoulder width in pixels
            hip_width: Hip width in pixels
            
        Returns:
            Waist definition as a ratio (lower = more defined waist)
        """
        # Use the larger of shoulder or hip width as reference
        reference_width = max(shoulder_width, hip_width)
        
        if reference_width == 0:
            return 1.0
        
        return waist_width / reference_width
    
    def _classify_inverted_triangle(
        self,
        ratio_sh: float,
        waist_definition: float
    ) -> bool:
        """
        Check if body shape is Inverted Triangle.
        
        Inverted Triangle: Shoulders are significantly wider than hips.
        
        Args:
            ratio_sh: Shoulder-to-hip ratio
            waist_definition: Waist definition ratio
            
        Returns:
            True if shape is Inverted Triangle
        """
        return ratio_sh > self.RATIO_THRESHOLD_HIGH
    
    def _classify_triangle_pear(
        self,
        ratio_sh: float,
        waist_definition: float
    ) -> bool:
        """
        Check if body shape is Triangle (Pear).
        
        Triangle/Pear: Hips are significantly wider than shoulders.
        
        Args:
            ratio_sh: Shoulder-to-hip ratio
            waist_definition: Waist definition ratio
            
        Returns:
            True if shape is Triangle/Pear
        """
        return ratio_sh < self.RATIO_THRESHOLD_LOW
    
    def _classify_hourglass(
        self,
        ratio_sh: float,
        waist_definition: float
    ) -> bool:
        """
        Check if body shape is Hourglass.
        
        Hourglass: Shoulders and hips are roughly equal, with a defined waist.
        
        Args:
            ratio_sh: Shoulder-to-hip ratio
            waist_definition: Waist definition ratio
            
        Returns:
            True if shape is Hourglass
        """
        # Shoulders and hips are balanced
        balanced = self.RATIO_THRESHOLD_LOW <= ratio_sh <= self.RATIO_THRESHOLD_HIGH
        
        # Waist is significantly smaller than hips/shoulders
        defined_waist = waist_definition < self.HOURGLASS_WAIST_RATIO
        
        return balanced and defined_waist
    
    def _classify_rectangle(
        self,
        ratio_sh: float,
        waist_definition: float
    ) -> bool:
        """
        Check if body shape is Rectangle.
        
        Rectangle: Shoulders and hips are roughly equal, with minimal waist definition.
        
        Args:
            ratio_sh: Shoulder-to-hip ratio
            waist_definition: Waist definition ratio
            
        Returns:
            True if shape is Rectangle
        """
        # Shoulders and hips are balanced
        balanced = self.RATIO_THRESHOLD_LOW <= ratio_sh <= self.RATIO_THRESHOLD_HIGH
        
        # Waist is not significantly smaller (minimal definition)
        minimal_waist = waist_definition >= self.HOURGLASS_WAIST_RATIO
        
        return balanced and minimal_waist
    
    def classify(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Classify body shape based on extracted metrics.
        
        Args:
            metrics: Dictionary containing:
                - shoulder_px: Shoulder width in pixels
                - hip_px: Hip width in pixels
                - waist_px: Waist width in pixels
                
        Returns:
            Body shape classification string:
                - "Inverted Triangle": Shoulders wider than hips
                - "Triangle" or "Pear": Hips wider than shoulders
                - "Hourglass": Balanced with defined waist
                - "Rectangle": Balanced with minimal waist definition
            Returns None if metrics are invalid
        """
        # Validate input metrics
        if not all(key in metrics for key in ['shoulder_px', 'hip_px', 'waist_px']):
            return None
        
        shoulder_width = metrics['shoulder_px']
        hip_width = metrics['hip_px']
        waist_width = metrics['waist_px']
        
        # Check for valid measurements
        if shoulder_width <= 0 or hip_width <= 0 or waist_width <= 0:
            return None
        
        # Calculate ratios
        ratio_sh = self._calculate_shoulder_hip_ratio(shoulder_width, hip_width)
        waist_definition = self._calculate_waist_definition(
            waist_width, 
            shoulder_width, 
            hip_width
        )
        
        # Classify based on fashion industry standards
        # Order matters: Check most specific classifications first
        
        if self._classify_inverted_triangle(ratio_sh, waist_definition):
            return "Inverted Triangle"
        
        if self._classify_triangle_pear(ratio_sh, waist_definition):
            return "Pear"
        
        if self._classify_hourglass(ratio_sh, waist_definition):
            return "Hourglass"
        
        if self._classify_rectangle(ratio_sh, waist_definition):
            return "Rectangle"
        
        # Default fallback (should rarely happen with proper thresholds)
        return "Rectangle"
    
    def get_detailed_classification(
        self, 
        metrics: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed classification with ratios and metrics.
        
        Args:
            metrics: Dictionary containing body measurements
            
        Returns:
            Dictionary containing:
                - shape: Body shape classification
                - ratio_sh: Shoulder-to-hip ratio
                - waist_definition: Waist definition ratio
                - measurements: Original measurements
            Returns None if classification fails
        """
        shape = self.classify(metrics)
        
        if shape is None:
            return None
        
        shoulder_width = metrics['shoulder_px']
        hip_width = metrics['hip_px']
        waist_width = metrics['waist_px']
        
        ratio_sh = self._calculate_shoulder_hip_ratio(shoulder_width, hip_width)
        waist_definition = self._calculate_waist_definition(
            waist_width,
            shoulder_width,
            hip_width
        )
        
        return {
            'shape': shape,
            'ratio_sh': float(ratio_sh),
            'waist_definition': float(waist_definition),
            'measurements': {
                'shoulder_px': float(shoulder_width),
                'hip_px': float(hip_width),
                'waist_px': float(waist_width)
            }
        }
