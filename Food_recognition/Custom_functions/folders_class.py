import os
import shutil
class Folders:
    """ class to handle the folders needed in image recognition problems for the training and validation"""
    def __init__(self, directories, items = [""]):
        self.directories = directories
        self.items = items 
    def create_folders(self, path_base_dir = "./Data/traning_test"):
        """ this function creates a folder for every item and directory of the class"""
        for i in self.directories:
            for j in self.items:
                path =  os.path.join(path_base_dir,i)
                path = os.path.join(path,j)
                if not os.path.exists(path):
                    os.makedirs(path)    
        self.path_base_dir = path_base_dir
    def select_images(self, partition, path_original_dataset = "./Data/original_dataset"):
        """this function move the photos specify in particion to their destination folders.
           In this way, we can divide the data in training, validation and test."""
        for i in range(len(partition)):
            for j in self.items:
                path_to_copy = os.path.join(path_original_dataset,j)
                files = os.listdir(path_to_copy)
                try:
                    files.remove('.ipynb_checkpoints')
                except:
                    pass
                files = files[partition[i][0]:partition[i][1]]
                path_to_past = os.path.join(self.path_base_dir, self.directories[i])
                path_to_past = os.path.join(path_to_past, j)
                for k in files:
                    path_file= os.path.join(path_to_past,k)
                    file = os.path.join(path_to_copy,k)
                    shutil.copyfile(file, path_file)
                