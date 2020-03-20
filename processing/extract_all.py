import zipfile

path = "../material/faces.zip"
zip_object = zipfile.ZipFile(file=path, mode="r")
zip_object.extractall("../material")
