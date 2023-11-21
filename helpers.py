import requests
import json
#import pandas as pd
import tqdm as tqdm

def get_af_json(structure_id):
    '''
    Inputs: 
    structure_id      a uniprot id <str>, ie "Q08253"
    
    Returns:          <dict> of uniprot parameters for id
    '''
    res = requests.get('https://alphafold.ebi.ac.uk/api/prediction/'+
                     structure_id +
                     '?key=AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94').text

    json_res = json.loads(res)
    return json_res[0]  # Unlist list of 1, return dict

def get_uniprot_json(uniprot_proteome):
    '''
    Inputs: 
    uniprot_proteome      a uniprot proteome <str>, ie "UP000000625" (E. Coli)
    
    Returns:
    json_res[0]     a <dict> of uniprot parameters for id
    '''
    res = requests.get('https://rest.uniprot.org/uniprotkb/stream?format=json&query=%28proteome%3A' +
                     uniprot_proteome +
                     '%29').text

    json_res = json.loads(res)
    return json_res


def get_uniprot_csv(infile):
    '''
    Inputs: 
    infile      a file path <str>, ie "~/compute/uniprot_"
    
    Returns:
    json_res[0]     a <dict> of uniprot parameters for id
    '''
    return


q = get_af_json("Q08253")
r = get_uniprot_json("UP000000625")

print(list(q.items())[:200])
print(list(r.items())[:200])

'''
def build_captions_dict(proteome):
    captions_dict = {}
    
    url = 'https://rest.uniprot.org/uniprotkb/stream?format=txt&query=%28proteome%3A' + proteome + '%29'
    print(url)
    s = requests.Session()
    vals = ["AlphaFoldDB", "-!- FUNCTION"]
    protid = 0
    protfun = 0

    with s.get(url, headers={'content-type': 'html/text'}, stream=True) as resp:
        for line in resp.iter_lines():
            line = line.decode("utf8")
            entrycount = 0  # reset protein entry counter
            annotcount = 0  # reset annotation counter
            if "AlphaFoldDB" in line:
                protid = line.split(sep=";")[1].strip(" ")
                entrycount += 1
            while "-!- FUNCTION" in line:
                protfun = line.split(sep=":")[1]
                annotcount += 1
                
            if entrycount > 0 and annotcount > 0:
                captions_dict[protid] = protfun  # Write an annotated entry
            else:
                captions_dict[protid] = "Unannotated in UniProt"

    return captions_dict

e_coli_dict = build_captions_dict("UP000000625")

print(e_coli_dict["P03023"])
'''