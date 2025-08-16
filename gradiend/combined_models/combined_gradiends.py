import json
import os
import torch
import torch.nn as nn

from gradiend.training.gradiend_training import train
from gradiend.model import GradiendModel, LargeLinear, ModelWithGradiend


#TODO a LOT of cleaning it up, currently akin to a scaffolding

class StackedGradiend(GradiendModel):
    def __init__(self, input_dim, layers, models, latent_dim=1,*args, **kwargs):
        super().__init__(input_dim=input_dim, latent_dim=1, layers=layers, *args, **kwargs)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for model in models: 
            self.encoders.append(model.gradiend.encoder)
            self.decoders.append(model.gradiend.decoder)
        
        #in any case freeze encoder weights I do not want to retrain the encoder part, only the decoder if needed. 
        for encoder in self.encoders: 
            for param in encoder.parameters():
                param.requires_grad = False

        #TODO find a better way to 
        self.source = models[0].gradiend.kwargs['training']['source']
        #TODO find a better way to do this..
        self.base_model = models[0].base_model

        self.models = models
        self.device = self.base_model.device
    
    #also name encode? it's not really forward... 
    # but now the decoders should definitely be 'replaced' as well 
    def encode(self, x):
        encoded = []
        for encoder in self.encoders:
            out = encoder(x).item()
            encoded.append(out)
        return encoded 
    
    def decode(self, x):
        # from the deocoder I want ONE iutput... 
        # 'sum' or average'
        decoded = []
        for decoder in self.decoders:
            out = decoder(out)
            decoded.append(out)
        return decoded
    
    def modify_model_decode(self,x):
        decoded = []
        for decoder in self.decoders:
            out = decoder(torch.tensor([x], dtype=torch.float, device=self.device))
            decoded.append(out)
        return torch.mean(torch.stack(decoded), dim=0)
      
    def modify_model_decode_v1(self,x):
        decoded = []
        for decoder in self.decoders:
            out = decoder(torch.tensor([x], dtype=torch.float, device=self.device))
            decoded.append(out)
        return decoded 

    def forward(self, x, return_encoded=False):
        pass
        'TODO yeah i do not want to call this...'
        # how can i make sure that the right encoder-> decoders are being pared i mean they're being added to the lists appropriately..
        

    def extract_gradients(self, bert, return_dict=False):
        return super().extract_gradients(self.base_model, return_dict)


class CominedEncoderDecoder(GradiendModel): 
    def __init__(self, grad_models, num_encoders, input_dim, latent_dim, layers, freeze_encoder = False, activation='tanh',  dec_init= 'trained', enc_init=False, shared=False, decoder_factor = 1.0, **kwargs):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, layers=layers, activation=activation, **kwargs)

    
        self.num_encoders = num_encoders
        self.dec_init = dec_init
        self.freeze_encoder = freeze_encoder
        self.decoder_factor = decoder_factor
        self.latent_dim = latent_dim
        self.activation = activation
        self.shared = shared

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        if enc_init == False:
            if grad_models:
                self.encoder = nn.ModuleList()
                self.og_decoders = nn.ModuleList()
            else:
                self.encoder = nn.ModuleList([nn.Sequential(LargeLinear(input_dim, latent_dim, device=self.device), nn.Tanh()) for _ in range(num_encoders)])
                self.og_decoders = nn.ModuleList([nn.Sequential(LargeLinear(latent_dim, input_dim, device=self.device), nn.Tanh()) for _ in range(num_encoders)])
            
            if grad_models:
                for model in grad_models: 
                    self.encoder.append(model.gradiend.encoder)
                    self.og_decoders.append(model.gradiend.decoder)
        else: 
            self.encoder = nn.Sequential(LargeLinear(input_dim, num_encoders, device=self.device), nn.Tanh())

            if grad_models:
                self.og_decoders = nn.ModuleList() 
                self.og_encoders = nn.ModuleList()

                for model in grad_models: 
                    self.og_decoders.append(model.gradiend.decoder)
                    self.og_encoders.append(model.gradiend.encoder)
                    
                
                with torch.no_grad(): 
                    encoder_states = []
                    for encoder in self.og_encoders: 
                        encoder_states.append(encoder.state_dict())

                
                    for i, state in enumerate(encoder_states):
                        print(f"Encoder {i}:")
                        for name, tensor in state.items():
                            print(f"  {name}: {tensor.shape}")


                    new_state =  {}
                    state_dict = grad_models[0].gradiend.encoder.state_dict()
                    state_dict_1 = grad_models[1].gradiend.encoder.state_dict()
                    state_dict_2 = grad_models[2].gradiend.encoder.state_dict()

                    print(state_dict.keys(), state_dict_1.keys(), state_dict_2.keys())

                    for key in state_dict.keys():
                        if 'linear' in key and 'weight' in key:
                            new_state[key] = torch.cat([encoder_state[key] for encoder_state in encoder_states], dim=0)
                        elif 'linear' in key and 'bias' in key:
                            bias_avg = sum(encoder_state[key] for encoder_state in encoder_states) / len(encoder_states)
                            new_bias = torch.full((3,), bias_avg.item())

                            new_state[key] = new_bias.clone()

                    
                self.encoder.load_state_dict(new_state)
            else:
                self.og_decoders = nn.ModuleList([nn.Sequential(LargeLinear(latent_dim, input_dim, device=self.device), nn.Tanh()) for _ in range(num_encoders)])
                self.og_encoders = nn.ModuleList([nn.Sequential(LargeLinear(input_dim, latent_dim, device=self.device), nn.Tanh()) for _ in range(num_encoders)])



        if self.freeze_encoder: 
            for encoder in self.encoder: 
                for param in encoder.parameters():
                    param.requires_grad = False 
        
        self.decoder = nn.Sequential(LargeLinear(num_encoders, input_dim, device=self.device), nn.Tanh())

        self.encoder_scales = []

        # intialises the new decoder with the weights of the already trained decoders. 
        if dec_init == 'trained': 
            with torch.no_grad(): 
                decoder_states = []
                for decoder in self.og_decoders: 
                    decoder_states.append(decoder.state_dict())

            
                for i, state in enumerate(decoder_states):
                    print(f"Decoder {i}:")
                    for name, tensor in state.items():
                        print(f"  {name}: {tensor.shape}")
   

                new_state =  {}
                state_dict = self.og_decoders[0].state_dict()
                state_dict_1 = self.og_decoders[0].state_dict()
                state_dict_2 = self.og_decoders[0].state_dict()

                print(state_dict.keys(), state_dict_1.keys(), state_dict_2.keys())

                for key in state_dict.keys():
                    if 'linear' in key and 'weight' in key:
                        new_state[key] = torch.cat([decoder_state[key] for decoder_state in decoder_states], dim=1)
                    elif 'linear' in key and 'bias' in key:
                        new_state[key] = sum(decoder_state[key] for decoder_state in decoder_states) / len(decoder_states)

                # for key in state_dict.keys(): 
                #     if 'weight' in key: 
                #         new_state[key] = torch.cat([decoder_state[key] for decoder_state in decoder_states], dim=1) 

                #     if 'bias' in key: 
                #         new_state[key] = sum(decoder_state[key] for decoder_state in decoder_states) / len(decoder_states)

            self.decoder.load_state_dict(new_state)

        elif dec_init == 'scratch': 
            # for i in range(num_encoders): 
            #     x = self.encoder[i][0].weight.max().item() * self.decoder_factor
            #     self.encoder_scales.append(x)

            with torch.no_grad():
                nn.init.xavier_uniform_(self.decoder[0].weight)
                # for i, scale in enumerate(self.encoder_scales):
                #     nn.init.uniform_(self.decoder[0].weight[:, i:i+1], -scale, scale)
        # else: 
        #     NotImplementedError

    @property
    def encoder_norm(self):
        total_norm = 0.0
        if self.shared:
            for encoder in self.encoder:  
                total_norm += torch.norm(encoder[0].weight, p=2).item() ** 2
                return total_norm ** 0.5  
        else: 
            return torch.norm(self.encoder[0].weight, p=2).item()
    
    @property   
    def decoder_norm(self):
        return torch.norm(self.decoder[0].weight, p=2).item()
    


    @classmethod
    def from_pretrained(cls, load_path, device=None):
        # Load saved state dict
        model_path = os.path.join(load_path, 'pytorch_model.bin')
        config_path = os.path.join(load_path, 'config.json')
        checkpoint = torch.load(model_path, map_location=device)
        config_path = os.path.join(load_path, 'config.json')

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Rebuild architecture using saved metadata
        model = cls(
        **config,
        grad_models=[],  
        num_encoders=3,
        dec_init=None,
        enc_init= True  
        )

        # Load entire saved state
        # model.load_state_dict(checkpoint['state_dict'])

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

        model.name_or_path = load_path

        return model

    
    # @classmethod
    # def from_pretrained(cls, load_directory, device_encoder=None, device_decoder=None):
         
    #     model_path = os.path.join(load_directory, 'pytorch_model.bin')
    #     config_path = os.path.join(load_directory, 'config.json')

    #     # Load model configuration
    #     with open(config_path, 'r') as f:
    #         config = json.load(f)


    #     if 'llama' in load_directory.lower() and device_encoder is None and device_decoder is None:
    #         # check that two GPUs are available
    #         if torch.cuda.device_count() < 2:
    #             raise RuntimeError("Two GPUs are required for GRADIEND Llama models.")

    #         device_encoder = torch.device("cuda:1")
    #         device_decoder = torch.device("cuda:0")

    #     # Instantiate the model
    #     model = cls(**config, device_encoder=device_encoder, device_decoder=device_decoder, grad_models = None, num_encoders=3)

    # # todo check GPU?
    #     # Load model state dictionary
    #     state_dict = torch.load(model_path, map_location=device_decoder, weights_only=True)

    #     # Check if the model is a legacy checkpoint
    #     if 'encoder.0.weight' in state_dict and 'decoder.0.weight' in state_dict:
    #         state_dict = cls._load_legacy_state_dict(state_dict)

    #     model.load_state_dict(state_dict)

    #     model.name_or_path = load_directory

    #     if 'layers_path' in config:
    #         layers_path = os.path.join(load_directory, config['layers_path'])
    #         # Load sparse layers
    #         try:
    #             model.layers = torch.load(layers_path)
    #         except FileNotFoundError:
    #             print(f"Warning: {layers_path} not found. Using all layers by default. This will be deprecated soon. Please do only specify layers_path in config if a layers_path exists")

    #     return model

    def encode(self, x, shared):
        if shared:
            encoded = []
            for enc in self.encoder:
                out = enc(x).item()
                encoded.append(out)
            return encoded 
        else:
            return self.encoder(x)

    def modified_decode(self, x, method='sum'): 
        # x-is three dim  there is a diff between [1,1,1] and [1][1][1]...
        out = self.decoder(torch.tensor(x, dtype=torch.float, device=self.device))
        
        if method == 'sum': 
            return out.squeeze(0)
        else: 
            return torch.mean(out, dim=1)
        
        # this accepts a outputdim by 3 vector and should deliver a 1, input-dim vecotr for model modification 
        

    def forward(self, x, return_encoded=False):
        # possibly new implementation..... that allows for retraining of the Combined
        return super().forward(x, return_encoded)




import yaml
from gradiend.combined_models.combined_gradiends import StackedGradiend, CominedEncoderDecoder


if __name__ == '__main__': 
    config = yaml.safe_load(open("config.yml"))['M_F_N_leipzig']

    MF = "results/experiments/gradiend/MF/1e-05/bert-base-german-cased/0"
    MN = "results/experiments/gradiend/MN/1e-05/bert-base-german-cased/0"
    FN = "results/experiments/gradiend/FN/1e-5/bert-base-german-cased/0"

    MF_model = ModelWithGradiend.from_pretrained(MF)
    base_model = MF_model.base_model
    tokenizer = MF_model.tokenizer
    MN_model = ModelWithGradiend.from_pretrained(MN)
    FN_model = ModelWithGradiend.from_pretrained(FN)


    layer_map = {k: v for k, v in base_model.named_parameters() if 'cls.prediction' not in k.lower()}
    layers = [layer for layer in layer_map]

    if isinstance(layers, dict):
        input_dim = sum([v.sum() for v in layers.values()])
    else:
        input_dim = sum([layer_map[layer].numel() for layer in layers])

    combined_enc_dec = CominedEncoderDecoder(grad_models=[MF_model, MN_model, FN_model], num_encoders=3, input_dim=input_dim, latent_dim=1, layers = layers)
    combined_model_with_grad = ModelWithGradiend(base_model, combined_enc_dec, tokenizer)

    train(combined_model_with_grad,config=config, multi_task=False)


