class Batch:
    def __init__(self, ei,
            di = None, do = None,
            el = None, dl = None,
            am = None, epm = None, dpm = None):
        self.encoder_inputs = ei
        self.decoder_inputs = di
        self.decoder_outputs = do
        self.encoder_lengths = el
        self.decoder_lengths = dl
        self.attention_mask = am
        self.encoder_padding_mask = epm
        self.decoder_padding_mask = dpm

    def __len__(self):
        return self.encoder_inputs.shape[1]

    def cuda(self):
        self.encoder_inputs = self.encoder_inputs.cuda()

        if self.decoder_inputs is not None:
            self.decoder_inputs = self.decoder_inputs.cuda()

        if self.decoder_outputs is not None:
            self.decoder_outputs = self.decoder_outputs.cuda()

        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.cuda()

        if self.encoder_padding_mask is not None:
            self.encoder_padding_mask = self.encoder_padding_mask.cuda()

        if self.decoder_padding_mask is not None:
            self.decoder_padding_mask = self.decoder_padding_mask.cuda()

        return self

