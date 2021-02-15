import pickle
import json

def save_dico(name,dico):
    # f = open(name+'.pkl','ab')
    # pickle.dump(dico, f, pickle.HIGHEST_PROTOCOL)
    f = open(name+'.json', 'ab')
    json.dump(dico, f, indent=4)
    f.close()

def load_file_as_list_of_dico(name):
    with(open(name)) as f:
        data = json.load(f)
        return data

#debug function
def load_and_print(name):
    with(open(name,"rb")) as f:
        while True:
            try:
                print(pickle.load(f))
            except EOFError:
                break

