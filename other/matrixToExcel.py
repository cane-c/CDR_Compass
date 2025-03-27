import pandas as pd
import numpy as np


def matrixExel(expertMatrix_Int):
    # Ensure categories are sorted (A1, A2, ..., A26) based on the numeric part
    categories = sorted(expertMatrix_Int.keys(), key=lambda x: int(x[1:]))

    # Get all unique subcategory keys (e.g. EN1, EN2, S1, EC1, etc.)
    subcategories = sorted({key for cat in expertMatrix_Int.values() for key in cat.keys()})

    # Build a list of rows for the DataFrame.
    # For each category, each subcategory's cell will display "lower - upper".
    rows = []
    for cat in categories:
        row = {'Category': cat}
        for subcat in subcategories:
           cell = expertMatrix_Int[cat].get(subcat)
           if cell is not None:
              lower = cell.get('lower_fence')
              upper = cell.get('upper_fence')
              row[subcat] = f"{lower} - {upper}"
           else:
              row[subcat] = ""
        rows.append(row)

    # Create the DataFrame; 'Category' is the first column and each subcategory is one column.
    df = pd.DataFrame(rows)

    # Save the DataFrame to an Excel file
    df.to_excel("final_output_extremes.xlsx", index=False)



