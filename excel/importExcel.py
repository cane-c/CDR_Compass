import pandas as pd

#for direct rating
class ExpertEvaluation:
    def __init__(self, file_path):
        # Load the Excel file into a DataFrame
        self.data = pd.read_excel(file_path, index_col=0)
        
        # Prepare the dictionary structure
        self.evaluation_dict = self._create_evaluation_dict()

    def _create_evaluation_dict(self):
        evaluation_dict = {}
        
        # Iterate over the rows to build the dictionary
        for action_id, row in self.data.iterrows():
            # Convert each row into a dictionary with criteria as keys
            evaluation_dict[action_id] = row.to_dict()
        
        return evaluation_dict

    def get_evaluation(self):
        # Return the structured dictionary for this expert evaluation
        return self.evaluation_dict

#for interval rating
class Interval_ExpertEvaluation:
    def __init__(self, file_path):
        # Load the Excel file into a DataFrame
        self.data = pd.read_excel(file_path, index_col=0, dtype=str)  # Ensure data is read as strings
        
        # Prepare the dictionary structure
        self.evaluation_dict = self._create_evaluation_dict()

    def _parse_interval(self, value):
        """Convert a string like 'X-Y' into a list [X, Y]."""
        try:
            return list(map(int, value.split('-'))) if '-' in value else [int(value), int(value)]
        except ValueError:
            return None  # Handle possible incorrect formats gracefully

    def _create_evaluation_dict(self):
        evaluation_dict = {}
        
        # Iterate over the rows to build the dictionary
        for action_id, row in self.data.iterrows():
            # Convert each row into a dictionary with parsed interval values
            evaluation_dict[action_id] = {col: self._parse_interval(val) for col, val in row.items()}
        
        return evaluation_dict

    def get_evaluation(self):
        """Return the structured dictionary with intervals."""
        return self.evaluation_dict





