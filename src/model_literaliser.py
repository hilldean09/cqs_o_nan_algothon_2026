import base64
import io
import torch
import CQS_O_NaN as onan

policy = onan.MasterNeuralNet()
policy.load_state_dict( torch.load( "./model_save_25_i3", weights_only=True ) )

buffer = io.BytesIO()
torch.save( policy.state_dict(), buffer )
encoded_String = base64.b64encode( buffer.getvalue() ).decode( "ascii" )

with open( "model_weights_b64.txt", "w" ) as f:
    f.write( encoded_String )

print( "Encoded length:", len( encoded_String ), "characters" )
