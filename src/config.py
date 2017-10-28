
import ConfigParser,io,os

with open(os.path.dirname(__file__)+'\config.ini') as f:
    sample_config = f.read()
config = ConfigParser.RawConfigParser(allow_no_value=True)
config.readfp(io.BytesIO(sample_config))
lkh_lib=config.get('libs', 'lkh')
scags_lib=config.get('libs', 'scags')

def get_lkh_lib():
    return lkh_lib

def get_scags_lib():
    return scags_lib