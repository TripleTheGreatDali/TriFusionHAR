import gdown

# Corrected URL for direct download from Google Drive
test_data_down = 'https://drive.google.com/uc?id=1R3quDDc8WQsLSK9CdKySmPpQRHu2iRGY&export=download'

# Output file path
td_output_path = 'test.zip'

# Download the file
gdown.download(url=test_data_down, output=td_output_path, quiet=False)
