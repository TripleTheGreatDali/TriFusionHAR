import gdown

# URL of the file on Google Drive
trained_model    = 'https://drive.google.com/uc?id=1UvZ0ccxQHz4SOnm-VKNCWitiAtkRic-T'


# Output file path
tm_output_path = 'TriFusion_ucf101_100_epoch.pth'


# Download the file
gdown.download(trained_model, tm_output_path, quiet=False)
