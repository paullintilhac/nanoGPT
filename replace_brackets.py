import os
file_list = ["dyck-clean-dev","dyck-clean-val","dyck-clean-train","dyck-corrupted-dev","dyck-corrupted-val","dyck-corrupted-train"]

new_file_list = []
for in_file in file_list:      
    f = open("data/"+in_file+".txt")

    fh = "data/"+ in_file+"-rep.txt"
    try:
        os.remove(fh)
    except OSError:
        pass
    print("old file deleted")
    new_file_list +=fh
    with open(fh,'a') as new_file:
        for line in f:
            line = line.replace('(a', '(')
            line = line.replace('a)', ')')
            line = line.replace('(b', '{')
            line = line.replace('b)', '}')
            line = line.replace('(c', '<')
            line = line.replace('c)', '>')
            line = line.replace('(d', '[')
            line = line.replace('d)', ']')
            new_file.write(line)
            # how to write this newline back to the file
