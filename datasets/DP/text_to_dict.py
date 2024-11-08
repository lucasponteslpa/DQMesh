

def text_to_dict(text_file_path):
    # Open the text file and read the lines
    try:
        with open(text_file_path, 'r') as text_file:
            lines = text_file.readlines()

        # Remove any extra newlines or leading/trailing spaces from each line
        params_dict = {
            'orientation':[],
            'position':[],
            'principal_point':[],
            'radial_distortion':[]
            }
        for line in lines:
            param = line.strip().split(': ')
            if param[0] in ['orientation','position','principal_point','radial_distortion']:
                params_dict[param[0]] += [float(param[1])]
            elif param[1] != 'PERSPECTIVE':
                params_dict[param[0]] = float(param[1])
        # lines = [line.strip().split(':') for line in lines]

        # Create a list of lines for the JSON data
        # json_data = {
        #     "lines": lines
        # }
        return params_dict

    except FileNotFoundError:
        print(f"Error: The file '{text_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
