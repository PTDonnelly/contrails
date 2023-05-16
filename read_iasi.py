from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatJsonRenderer
from pybufrkit.encoder import Encoder
from pybufrkit.decoder import generate_bufr_message
from pybufrkit.mdquery import MetadataQuerent, MetadataExprParser
from pybufrkit.dataquery import DataQuerent, NodePathParser
from pybufrkit.script import ScriptRunner

filename = "C:\\Users\\padra\\Documents\\Research\\data\\iasi\\W_XX-EUMETSAT-Darmstadt,SOUNDING+SATELLITE,METOPA+IASI_C_EUMC_20200101000253_68496_eps_o_l1.bin"

# Decode a BUFR file
decoder = Decoder()
with open(filename, 'rb') as ins:
    bufr_message = decoder.process(ins.read())

print('Decoded')

# Convert the BUFR message to JSON
json_data = FlatJsonRenderer().render(bufr_message)

print('Coverted to JSON')

# Encode the JSON back to BUFR file
encoder = Encoder()
bufr_message_new = encoder.process(json_data)
with open("bufr_output.txt", 'wb') as outs:
    outs.write(bufr_message_new.serialized_bytes)

print('Re-encoded')

# Decode for multiple messages from a single file
with open("bufr_output.txt", 'rb') as ins:
    for bufr_message in generate_bufr_message(decoder, ins.read()):
        pass  # do something with the decoded message object

print('Extra re-encoding')

# Query the metadata MetadataQuerent
n_subsets = MetadataQuerent(MetadataExprParser()).query(bufr_message, '%n_subsets')

print('Query metadata')

# Query the data DataQuerent
query_result = DataQuerent(NodePathParser()).query(bufr_message, '001002')
print(type(query_result))
print('Query data')

# Script
# NOTE: must use the function version of print (Python 3), NOT the statement version
code = """print('Multiple' if ${%n_subsets} > 1 else 'Single')"""
runner = ScriptRunner(code)
runner.run(bufr_message)