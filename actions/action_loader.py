"""
Action Library Loader
Dynamically loads action definitions from JSON and provides query utilities
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class ActionLibrary:
    """Loads and provides access to action library"""
    
    def __init__(self, library_path: str):
        """
        Load action library from JSON file
        
        Args:
            library_path: Path to action_library.json file
        """
        self.library_path = Path(library_path)
        
        if not self.library_path.exists():
            raise FileNotFoundError(f"Action library not found: {library_path}")
        
        # Load the library
        with open(self.library_path, 'r') as f:
            self.library_data = json.load(f)
        
        # Extract components
        self.actions = {action['id']: action for action in self.library_data['actions']}
        self.metadata = self.library_data.get('metadata', {})
        self.version = self.library_data.get('version', 'unknown')
        
        # Build category index
        self.categories = {}
        for action in self.library_data['actions']:
            category = action.get('category', 'uncategorized')
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(action['id'])
    
    def get_action(self, action_id: str) -> Optional[Dict]:
        """
        Get action definition by ID
        
        Args:
            action_id: Action identifier (e.g., 'location_setting')
            
        Returns:
            Action definition dict, or None if not found
        """
        return self.actions.get(action_id)
    
    def get_all_actions(self) -> List[Dict]:
        """
        Get all action definitions
        
        Returns:
            List of all action definitions
        """
        return list(self.actions.values())
    
    def get_actions_by_category(self, category: str) -> List[Dict]:
        """
        Get actions in a specific category
        
        Args:
            category: Category name (e.g., 'scene_level', 'temporal_environmental')
            
        Returns:
            List of action definitions in that category
        """
        action_ids = self.categories.get(category, [])
        return [self.actions[aid] for aid in action_ids]
    
    def get_parameter_enum_values(self, action_id: str, param_name: str) -> List[str]:
        """
        Get valid enum values for a parameter
        
        Args:
            action_id: Action identifier
            param_name: Parameter name
            
        Returns:
            List of valid enum values, or empty list if not enum or not found
        """
        action = self.get_action(action_id)
        if not action:
            return []
        
        params = action.get('parameters', {})
        param_def = params.get(param_name, {})
        
        if param_def.get('type') == 'enum':
            return param_def.get('values', [])
        
        return []
    
    def validate_action_plan(self, actions: List[Dict]) -> tuple[bool, str]:
        """
        Validate an action plan against schema
        
        Args:
            actions: List of action dicts with 'action_id' and 'parameters'
            
        Returns:
            (is_valid, error_message)
        """
        if not actions:
            return False, "Action plan is empty"
        
        max_actions = self.metadata.get('max_actions_per_plan', 5)
        if len(actions) > max_actions:
            return False, f"Too many actions: {len(actions)} (max: {max_actions})"
        
        for i, action in enumerate(actions, 1):
            action_id = action.get('action_id')
            if not action_id:
                return False, f"Action {i}: missing 'action_id'"
            
            action_def = self.get_action(action_id)
            if not action_def:
                return False, f"Action {i}: unknown action_id '{action_id}'"
            
            # Validate parameters
            params = action.get('parameters', {})
            param_defs = action_def.get('parameters', {})
            
            # Check required parameters
            for param_name, param_def in param_defs.items():
                if param_def.get('required', False) and param_name not in params:
                    return False, f"Action {i} ({action_id}): missing required parameter '{param_name}'"
            
            # Validate enum values
            for param_name, param_value in params.items():
                param_def = param_defs.get(param_name, {})
                if param_def.get('type') == 'enum':
                    valid_values = param_def.get('values', [])
                    if param_value not in valid_values:
                        return False, f"Action {i} ({action_id}): invalid value '{param_value}' for parameter '{param_name}'"
        
        return True, "Valid"
    
    def get_metadata(self) -> Dict:
        """
        Get library metadata (version, total_actions, etc.)
        
        Returns:
            Metadata dict
        """
        return self.metadata
    
    def get_version(self) -> str:
        """
        Get library version
        
        Returns:
            Version string
        """
        return self.version
    
    def __repr__(self):
        return f"ActionLibrary(version={self.version}, actions={len(self.actions)})"

