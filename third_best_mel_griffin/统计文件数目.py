import os

path2 = '../p225/'
dirs = os.listdir( path2 )
print("type(dirs) = ", type(dirs))  # type(dirs) =  <class 'list'>
print("len(dirs) = ", len(dirs))  # len(dirs) =  14

count = 0
for file in dirs:
   print( file)
   if '.wav' in file:
       count = count+1
       print("------------")
       print(file)
       print(file.replace('.wav', ''))
       print(file)
       print(type(file))
       print(file.encode())
       print("------------")
print("**********""")
print(len(dirs))

print(count)


