import os

# print(os.path.abspath(os.path.join(os.path.dirname('data'), '..', 'templates')))
os.chdir('..')
os.chdir('dlgo/data/data')
print(os.path.abspath(os.curdir))
# if not os.path.isdir(self.data_directory):
#    os.makedirs(self.data_directory)