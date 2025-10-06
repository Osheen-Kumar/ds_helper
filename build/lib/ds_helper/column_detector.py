import pandas as pd

def detect_column_types(df, unique_value_threshold=20): 
    #function to detect column types 
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    column_types = {}

    for column in df.columns:
    
        num_unique_values = df[column].nunique()

        
        dtype = df[column].dtype

        
        if pd.api.types.is_numeric_dtype(dtype):
            
            if num_unique_values < unique_value_threshold:
                column_types[column] = 'categorical'
            else:
                column_types[column] = 'numerical'

       
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
           
            if num_unique_values < unique_value_threshold:
                column_types[column] = 'categorical'
           
            else:
                column_types[column] = 'text'

        
        else:
            column_types[column] = 'text'

    return column_types