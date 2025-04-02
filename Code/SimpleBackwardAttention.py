import torch
from torch.cuda import is_available
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.utils
import warnings
import gc

CONFIG_DICT = {'vocab_size': 50257,
               'dim': {'in': 768,
                       'out': 30,
                       'latent': 28,
                       'hidden': 768},
               'vals': {'heads': 24,
                        'dropout': 0.1,
                        'eps': 1e-8,
                        'layers': 1,
                        'chunks': 29},
               'flags': {'in': False,
                         'out': False,
                         'latent': True,
                         'q_transform': False,
                         'head_transform': False,
                         'multihead': False,
                         'hidden': False,
                         'chunk': True,
                         'layer_transform': False,
                         'sequential': False},
               'bias': {'in': False,
                        'out': False,
                        'latent': False,
                        'head': False,
                        'q': False,
                        'q_proj': False,
                        'kv': False,
                        'hidden': False,
                        'chunk': False,
                        'layer': False},
               'device': ('cuda' if torch.cuda.is_available() else 'cpu'),
               'dtype': torch.bfloat16}


def device_match(x: torch.Tensor, device: str, dtype: torch.float32, supress_warnings: bool = False):
    if x.numel() == 0:
        print('No Tensor is passed')
        return torch.empty(0, device=torch.device(device), dtype=dtype)
    if x.device != torch.device(device):
        if not supress_warnings:
            warnings.warn(f'Expected the tensor on this device:{torch.device(device)} '
                          f'but got {x.device}\n Trying to cast it to same device', UserWarning)
        try:
            x = x.to(torch.device(device))
        except RuntimeError:
            raise RuntimeError('Failed to cast the tensor to same device')
    if x.dtype != dtype:
        if not supress_warnings:
            warnings.warn(f'Expected the tensor to be same as {dtype}'
                          f'but got {x.dtype}\n Trying to cast it to same dtype', UserWarning)
        try:
            x = x.to(dtype)
        except RuntimeError:
            raise RuntimeError('Failed to cast the tensor to same dtype')
    return x


class BackwardAttention(nn.Module):
    """
    :config = cfg = {'vocab_size':int,
                     'dim':{'in':int,
                            'out':int,
                            'latent':int,
                            'hidden':int},
                     'vals':{'heads':int,
                             'dropout':float,
                             'eps':float,
                             'layers':int,
                             'chunks':int},
                     'flags':{'in':bool,
                              'out':bool,
                              'latent':bool,
                              'q_transform':bool,
                              'head_transform':bool,
                              'multihead':bool,
                              'hidden':bool,
                              'chunk':bool},
                     'bias':{'in':bool,
                             'out':bool,
                             'latent':bool,
                             'head':bool,
                             'q':bool,
                             'q_proj':bool,
                             'kv':bool,
                             'hidden':bool,
                             'chunk':bool},
                     'device':str,
                     'dtype':torch.dtype}
    """

    def __init__(self, cfg: dict):
        super().__init__()
        if cfg['flags']['in']:
            self.W_in = nn.Linear(cfg['dim']['in'], cfg['dim']['in'], bias=cfg['bias']['in'],
                                  device=cfg['device'], dtype=cfg['dtype'])
        if cfg['flags']['latent']:
            self.Wl = nn.Linear(cfg['dim']['in'], cfg['dim']['latent'], bias=cfg['bias']['latent'],
                                device=cfg['device'], dtype=cfg['dtype'])
            self.Wk = nn.Linear(cfg['dim']['latent'], cfg['dim']['out'], bias=cfg['bias']['kv'],
                                device=cfg['device'], dtype=cfg['dtype'])
            self.Wv = nn.Linear(cfg['dim']['latent'], cfg['dim']['out'], bias=cfg['bias']['kv'],
                                device=cfg['device'], dtype=cfg['dtype'])
        else:
            self.Wk = nn.Linear(cfg['dim']['in'], cfg['dim']['out'], bias=cfg['bias']['kv'],
                                device=cfg['device'], dtype=cfg['dtype'])
            self.Wv = nn.Linear(cfg['dim']['in'], cfg['dim']['out'], bias=cfg['bias']['kv'],
                                device=cfg['device'], dtype=cfg['dtype'])
        if not cfg['flags']['multihead']:
            self.Wq = nn.Linear(cfg['dim']['out'], cfg['dim']['out'], bias=cfg['bias']['q'],
                                device=cfg['device'], dtype=cfg['dtype'])
        else:
            if cfg['dim']['out'] % cfg['vals']['heads'] != 0:
                raise ValueError(f"The dimensions {cfg['dim']['out']} must be divisible by {cfg['vals']['heads']}")
            cfg['dim']['head'] = cfg['dim']['out'] // cfg['vals']['heads']
            self.Wq = nn.Linear(cfg['dim']['head'], cfg['dim']['head'], bias=cfg['bias']['head'],
                                device=cfg['device'], dtype=cfg['dtype'])
            if cfg['flags']['head_transform']:
                self.h_expand = nn.Linear(1, cfg['vals']['heads'], bias=cfg['bias']['head'],
                                          device=cfg['device'], dtype=cfg['dtype'])
                self.h_compress = nn.Linear(cfg['vals']['heads'], 1, bias=cfg['bias']['head'],
                                            device=cfg['device'], dtype=cfg['dtype'])
        if cfg['flags']['hidden']:
            self.W_hidden = nn.Linear(cfg['dim']['hidden'], cfg['vocab_size'], bias=cfg['bias']['hidden'],
                                      device=cfg['device'], dtype=cfg['dtype'])
            self.var = nn.Linear(2 * cfg['vocab_size'], cfg['vocab_size'], bias=False,
                                 device=cfg['device'], dtype=cfg['dtype'])
        if cfg['flags']['out']:
            self.out_proj = nn.Linear(cfg['dim']['out'], cfg['dim']['out'], bias=cfg['bias']['out'],
                                      device=cfg['device'], dtype=cfg['dtype'])
        if cfg['flags']['q_transform']:
            if not cfg['flags']['chunk']:
                self.q_proj = nn.Linear(cfg['vocab_size'], 1, bias=cfg['bias']['q_proj'],
                                        device=cfg['device'], dtype=cfg['dtype'])
            else:
                if cfg['vocab_size'] % cfg['vals']['chunks'] != 0:
                    raise ValueError(f"The value {cfg['vocab_size']} must be divisible by {cfg['vals']['chunks']}")
                cfg['dim']['chunk'] = cfg['vocab_size'] // cfg['vals']['chunks']
                self.chunk_proj1 = nn.ModuleList([nn.Linear(cfg['dim']['chunk'], 1, bias=cfg['bias']['chunk'],
                                                            device=cfg['device'], dtype=cfg['dtype'])
                                                  for _ in range(cfg['vals']['chunks'])])
        self.cfg = cfg
        self.dropout = nn.Dropout(cfg['vals']['dropout'])

    def clear_var(self, *args):
        for arg in args:
            del arg
        gc.collect()
        if self.cfg['device'] == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def un_chunked_attn(self, embeddings: torch.Tensor, logits: torch.Tensor,
                        hidden_states=None, supress_warnings=False) -> torch.tensor:
        embeddings = device_match(embeddings, self.cfg['device'], self.cfg['dtype'], supress_warnings=supress_warnings)
        if hidden_states is None:
            hidden_states = torch.empty((0,), device=self.cfg['device'], dtype=self.cfg['dtype'])
        if self.cfg['flags']['in']:
            embeddings = self.W_in(embeddings)
        if self.cfg['flags']['latent']:
            L = self.Wl(embeddings)
            self.clear_var(embeddings)
        if len(logits.size()) == 3 and logits.size(-1) == self.cfg['vocab_size']:
            logits = device_match(logits[:, -1, :].squeeze(1), self.cfg['device'], self.cfg['dtype'],
                                  supress_warnings=supress_warnings)
        elif len(logits.size()) == 2 and logits.size(-1) == self.cfg['vocab_size']:
            logits = device_match(logits, self.cfg['device'], self.cfg['dtype'], supress_warnings=supress_warnings)
        elif len(logits.size()) == 1 and logits.size(0) == self.cfg['vocab_size']:
            logits = device_match(logits, self.cfg['device'], self.cfg['dtype'],
                                  supress_warnings=supress_warnings).unsqueeze(0)
        else:
            raise RuntimeError(f"Expected tensor of dimension 1 or 2 or 3 but got {len(logits.size())}")
        if self.cfg['flags']['hidden']:
            if len(hidden_states.size()) == 3 and hidden_states.size(-1) == self.cfg['dim']['hidden']:
                hidden_states = device_match(hidden_states[:, -1, :].squeeze(1), self.cfg['device'], self.cfg['device'],
                                             supress_warnings=supress_warnings)
            elif len(hidden_states.size()) == 2 and hidden_states.size(-1) == self.cfg['dim']['hidden']:
                hidden_states = device_match(hidden_states[-1, :].squeeze(1), self.cfg['device'],
                                             self.cfg['device'], supress_warnings=supress_warnings).unsqueeze(0)
            else:
                raise RuntimeError(f"Expected tensor of dimension 1 or 2 or 3 but got {len(hidden_states.size())}")
            hidden_states = self.W_hidden(hidden_states)
            logits = self.var(torch.cat([logits, hidden_states], dim=-1))
            self.clear_var(hidden_states)
        logits = F.softmax(logits, dim=-1)
        logits = self.dropout(logits)
        b, v = logits.size()
        if self.cfg['flags']['latent']:
            V = self.Wv(L)
        else:
            V = self.Wv(embeddings)  # type:ignore
        if not self.cfg['flags']['multihead']:
            vector = logits.unsqueeze(-1) * V
            self.clear_var(logits, V)
            if self.cfg['flags']['q_transform']:
                vector = self.q_proj(vector.transpose(-1, -2)).transpose(-1, -2) / self.cfg['vocab_size']
            else:
                vector = vector.sum(dim=-2, keepdim=True) / self.cfg['vocab_size']
            vector = self.Wq(vector)
            if self.cfg['flags']['latent']:
                K = self.Wk(L)
                self.clear_var(L)
            else:
                K = self.Wk(embeddings)  # type:ignore
                self.clear_var(embeddings)
            vector = torch.matmul(vector, K.transpose(0, 1)) / math.sqrt(K.size(-1))
            self.clear_var(K)
            if self.cfg['flags']['out']:
                vector = self.out_proj(vector)
            vector = vector.squeeze(1)
            return vector
        else:
            head_dim = self.cfg['dim']['head']
            heads = self.cfg['vals']['heads']
            if self.cfg['flags']['head_transform']:
                logits = self.h_expand(logits.unsqueeze(-1)).transpose(-1, -2)
            else:
                logits = logits.unsqueeze(-1).expand(-1, -1, heads).transpose(-1, -2)
            V = V.unsqueeze(0).expand(b, -1, -1).view(b, v, heads, head_dim).transpose(1, 2)
            vector = logits.unsqueeze(-1) * V
            self.clear_var(logits, V)
            if self.cfg['flags']['q_transform']:
                vector = self.q_proj(vector.transpose(-1, -2)).transpose(-1, -2) / self.cfg['vocab_size']
            else:
                vector = vector.sum(dim=-2, keepdim=True) / self.cfg['vocab_size']
            vector = self.Wq(vector)
            if self.cfg['flags']['latent']:
                K = self.Wk(L).unsqueeze(0).expand(b, -1, -1).view(b, v, heads, head_dim).transpose(1, 2)
                self.clear_var(L)
            else:
                K = self.Wk(embeddings).unsqueeze(0).expand(b, -1, -1).view(b, v, heads, head_dim).transpose(1,
                                                                                                             2)  # type:ignore
                self.clear_var(embeddings)
            vector = torch.matmul(vector, K.transpose(-1, -2)) / math.sqrt(head_dim)
            self.clear_var(K)
            vector = vector.squeeze(2).transpose(1, 2)
            if self.cfg['flags']['head_transform']:
                vector = self.h_compress(vector)
            else:
                vector = (vector.sum(dim=-1, keepdim=False))
            vector = vector.squeeze(-1)
            if self.cfg['flags']['out_proj']:
                vector = self.out_proj(vector)
            return vector

    def chunked_attn(self, embeddings: torch.Tensor, logits: torch.Tensor,
                     hidden_states=None, supress_warnings=False) -> torch.tensor:
        score = torch.empty((0,), device=self.cfg['device'], dtype=self.cfg['dtype'])
        if hidden_states is None:
            hidden_states = torch.empty((0,), device=self.cfg['device'], dtype=self.cfg['dtype'])
        if len(logits.size()) == 3:
            logits = device_match(logits[:, -1, :].squeeze(1), self.cfg['device'], self.cfg['dtype'],
                                  supress_warnings=supress_warnings)
        elif len(logits.size()) == 2:
            logits = device_match(logits, self.cfg['device'], self.cfg['dtype'], supress_warnings=supress_warnings)
        elif len(logits.size()) == 1:
            logits = device_match(logits, self.cfg['device'], self.cfg['dtype'],
                                  supress_warnings=supress_warnings).unsqueeze(0)
        else:
            raise RuntimeError(f"Expected tensor of dimension 1 or 2 or 3 but got {len(logits.size())}")
        if self.cfg['flags']['hidden']:
            if len(hidden_states.size()) == 3 and hidden_states.size(-1) == self.cfg['dim']['hidden']:
                hidden_states = device_match(hidden_states[:, -1, :].squeeze(1), self.cfg['device'],
                                             self.cfg['device'], supress_warnings=supress_warnings)
            elif len(hidden_states.size()) == 2 and hidden_states.size(-1) == self.cfg['dim']['hidden']:
                hidden_states = device_match(hidden_states[-1, :].squeeze(1), self.cfg['device'],
                                             self.cfg['device'], supress_warnings=supress_warnings).unsqueeze(0)
            else:
                raise RuntimeError(f"Expected tensor of dimension 1 or 2 or 3 but got {len(hidden_states.size())}")
        for i in range(self.cfg['vals']['chunks']):
            start = i * self.cfg['dim']['chunk']
            end = (i + 1) * self.cfg['dim']['chunk']
            emb = device_match(embeddings[start:end, :], self.cfg['device'], self.cfg['dtype'],
                               supress_warnings=supress_warnings)
            if self.cfg['flags']['in']:
                emb = self.W_in(emb)
            if self.cfg['flags']['latent']:
                L = self.Wl(emb)
                self.clear_var(emb)
            logs = logits[:, start:end]
            if self.cfg['flags']['hidden']:
                hidden_states = hidden_states[:, start:end]
                hidden_states = self.W_hidden(hidden_states)
                logs = self.var(torch.cat([logs, hidden_states], dim=-1))
                self.clear_var(hidden_states)
            logs = F.softmax(logs, dim=-1)
            logs = self.dropout(logs)
            b, v = logs.size()
            if self.cfg['flags']['latent']:
                V = self.Wv(L)
            else:
                V = self.Wv(emb)  # type:ignore                
            if not self.cfg['flags']['multihead']:
                vector = logs.unsqueeze(-1) * V
                self.clear_var(logs, V)
                if self.cfg['flags']['q_transform']:
                    vector = self.chunk_proj1[i](vector.transpose(-1, -2)).transpose(-1, -2) / self.cfg['vocab_size']
                else:
                    vector = vector.sum(dim=-2, keepdim=True) / self.cfg['vocab_size']
                vector = self.Wq(vector)
                if self.cfg['flags']['latent']:
                    K = self.Wk(L)
                    self.clear_var(L)
                else:
                    K = self.Wk(emb)  # type:ignore
                    self.clear_var(emb)
                vector = torch.matmul(vector, K.transpose(0, 1)) / math.sqrt(K.size(-1))
                self.clear_var(K)
                vector = vector.squeeze(-2)
            else:
                head_dim = self.cfg['dim']['head']
                heads = self.cfg['vals']['heads']
                if self.cfg['flags']['head_transform']:
                    logs = self.h_expand(logits.unsqueeze(-1)).transpose(-1, -2)
                else:
                    logs = logits.unsqueeze(-1).expand(-1, -1, heads).transpose(-1, -2)
                V = V.unsqueeze(0).expand(b, -1, -1).view(b, v, heads, head_dim).transpose(1, 2)
                vector = logs.unsqueeze(-1) * V
                self.clear_var(logs, V)
                if self.cfg['flags']['q_transform']:
                    vector = self.chunk_proj1[i](vector.transpose(-1, -2)).transpose(-1, -2) / self.cfg['vocab_size']
                else:
                    vector = vector.sum(dim=-2, keepdim=True) / self.cfg['vocab_size']
                vector = self.Wq(vector)
                if self.cfg['flags']['latent']:
                    K = self.Wk(L).unsqueeze(0).expand(b, -1, -1).view(b, v, heads, head_dim).transpose(1, 2)
                    self.clear_var(L)
                else:
                    K = self.Wk(emb).unsqueeze(0).expand(b, -1, -1).view(b, v, heads, head_dim).transpose(1,
                                                                                                          2)  # type:ignore
                    self.clear_var(emb)
                vector = self.Wq(vector)
                vector = torch.matmul(vector, K.transpose(-1, -2)) / math.sqrt(head_dim)
                self.clear_var(K)
                vector = vector.squeeze(2).transpose(1, 2)
                if self.cfg['flags']['head_transform']:
                    vector = self.h_compress(vector)
                else:
                    vector = (vector.sum(dim=-1, keepdim=False))
                vector = vector.squeeze(-1)
            if score.numel() == 0:
                score = vector
                self.clear_var(vector)
            else:
                score = torch.cat([score, vector], dim=-1)
                self.clear_var(vector)
        if self.cfg['flags']['out']:
            score = self.out_proj(score)
        return score

    def forward(self, embeddings: torch.Tensor, logits: torch.Tensor,
                hidden_states=None, supress_warnings=False) -> torch.tensor:
        if hidden_states is None:
            hidden_states = torch.empty((0,), device=self.cfg['device'], dtype=self.cfg['dtype'])
        if not self.cfg['flags']['chunk']:
            return self.un_chunked_attn(embeddings=embeddings, logits=logits,
                                        hidden_states=hidden_states, supress_warnings=supress_warnings)
        else:
            return self.chunked_attn(embeddings=embeddings, logits=logits,
                                     hidden_states=hidden_states, supress_warnings=supress_warnings)


class SimpleBackwardAttention(nn.Module):
    """
        :config = cfg = {'vocab_size':int,
                         'dim':{'in':int,
                                'out':int,
                                'latent':int,
                                'hidden':int},
                         'vals':{'heads':int,
                                 'dropout':float,
                                 'eps':float,
                                 'layers':int,
                                 'chunks':int},
                         'flags':{'in':bool,
                                  'out':bool,
                                  'latent':bool,
                                  'q_transform':bool,
                                  'head_transform':bool,
                                  'multihead':bool,
                                  'hidden':bool,
                                  'chunk':bool,
                                  'sequential':bool},
                         'bias':{'in':bool,
                                 'out':bool,
                                 'latent':bool,
                                 'head':bool,
                                 'q':bool,
                                 'q_proj':bool,
                                 'kv':bool,
                                 'hidden':bool,
                                 'chunk':bool,
                                 'layer':bool},
                         'device':str,
                         'dtype':torch.dtype}
        """

    def __init__(self, cfg):
        super().__init__()
        attn_cfg = {'vocab_size': cfg['vocab_size'],
                    'dim': {'in': cfg['dim']['in'],
                            'out': cfg['dim']['out'],
                            'latent': cfg['dim']['latent'],
                            'hidden': cfg['dim']['hidden']},
                    'vals': {'heads': cfg['vals']['heads'],
                             'dropout': cfg['vals']['dropout'],
                             'chunks': cfg['vals']['chunks']},
                    'flags': {'in': cfg['flags']['in'],
                              'out': cfg['flags']['out'],
                              'latent': cfg['flags']['latent'],
                              'q_transform': cfg['flags']['q_transform'],
                              'head_transform': cfg['flags']['head_transform'],
                              'multihead': cfg['flags']['multihead'],
                              'hidden': cfg['flags']['hidden'],
                              'chunk': cfg['flags']['chunk']},
                    'bias': {'in': cfg['bias']['in'],
                             'out': cfg['bias']['out'],
                             'latent': cfg['bias']['latent'],
                             'head': cfg['bias']['head'],
                             'q': cfg['bias']['q'],
                             'kv': cfg['bias']['kv'],
                             'hidden': cfg['bias']['hidden'],
                             'chunk': cfg['bias']['chunk']},
                    'device': cfg['device'],
                    'dtype': cfg['dtype']}
        self.attn = nn.ModuleList([BackwardAttention(attn_cfg)
                                   for _ in range(cfg['vals']['layers'])])
        if cfg['vals']['layers'] < 0:
            raise ValueError(f"The given object cannot be created for the give value {cfg['vals']['layers']}")
        elif cfg['vals']['layers'] == 1:
            self.attention = BackwardAttention(cfg)
        if cfg['vocab_size'] % cfg['vals']['chunks'] != 0:
            raise ValueError(f"The value {cfg['vocab_size']} must be divisible by {cfg['vals']['chunks']}")
        cfg['dim']['chunk'] = cfg['vocab_size'] // cfg['vals']['chunks']
        self.dropout = nn.Dropout(cfg['vals']['dropout'])
        self.weight = nn.Parameter(torch.ones(cfg['dim']['in'], device=cfg['device'], dtype=cfg['dtype']))
        if not cfg['flags']['sequential']:
            self.compress = nn.Linear(cfg['vals']['layers'], 1, bias=cfg['bias']['layer'],
                                      device=cfg['device'], dtype=cfg['dtype'])
        self.cfg = cfg

    def _chunked_emb_norm(self, emb: torch.Tensor, supress_warnings=False) -> torch.tensor:
        for i in range(self.cfg['vals']['chunks']):
            start = i * self.cfg['dim']['chunk']
            end = (i + 1) * self.cfg['dim']['chunk']
            chunk = device_match(emb[start:end, :], self.cfg['device'], self.cfg['dtype'],
                                 supress_warnings=supress_warnings)
            rms = chunk.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.cfg['vals']['eps']
            yield chunk / rms
            del chunk, rms
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def emb_norm(self, emb: torch.Tensor, supress_warnings=False) -> torch.Tensor:
        if not self.cfg['flags']['chunk']:
            emb = device_match(emb, self.cfg['device'], self.cfg['dtype'], supress_warnings=supress_warnings)
            rms = emb.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.cfg['vals']['eps']
            emb /= rms
            return emb * self.weight
        else:
            chunk_list = []
            for chunk in self._chunked_emb_norm(emb):
                chunk = device_match(chunk, self.cfg['device'], self.cfg['dtype'], supress_warnings=True)
                chunk_list.append(chunk)
            chunk_list = torch.cat(chunk_list, dim=0)

            return self.weight * chunk_list

    def logits_norm(self, logits: torch.Tensor, supress_warnings=False) -> torch.Tensor:
        logits = device_match(logits, self.cfg['device'], self.cfg['dtype'], supress_warnings=supress_warnings)
        logits /= logits.pow(2).mean(dim=-1, keepdim=True).sqrt() + self.cfg['vals']['eps']
        return logits

    def forward(self, embeddings: torch.Tensor, logits: torch.Tensor,
                hidden_states=None, supress_warnings=False) -> torch.tensor:
        if hidden_states is None:
            hidden_states = torch.empty((0,), device=self.cfg['device'], dtype=self.cfg['dtype'])
        device = embeddings.device
        embeddings = self.emb_norm(embeddings + self.dropout(embeddings), supress_warnings=supress_warnings)
        embeddings.to(device)
        if len(logits.size()) == 3 and logits.size(-1) == self.cfg['vocab_size']:
            logits = device_match(logits[:, -1, :].squeeze(1), self.cfg['device'], self.cfg['dtype'],
                                  supress_warnings=supress_warnings)
        elif len(logits.size()) == 2 and logits.size(-1) == self.cfg['vocab_size']:
            logits = device_match(logits, self.cfg['device'], self.cfg['dtype'], supress_warnings=supress_warnings)
        elif len(logits.size()) == 1 and logits.size(0) == self.cfg['vocab_size']:
            logits = device_match(logits, self.cfg['device'], self.cfg['dtype'],
                                  supress_warnings=supress_warnings).unsqueeze(0)
        else:
            raise RuntimeError(f"Expected tensor of dimension 1 or 2 or 3 but got {len(logits.size())}")
        if self.cfg['vals']['layers'] == 1:
            device = embeddings.device
            embeddings = device_match(embeddings, self.cfg['device'], self.cfg['dtype'],
                                      supress_warnings=supress_warnings)
            embeddings = self.emb_norm(embeddings + self.dropout(embeddings)).to(device)
            logits += self.dropout(
                self.attention(embeddings, self.logits_norm(logits, supress_warnings=supress_warnings), hidden_states,
                               supress_warnings=supress_warnings))
            return F.softmax(logits, dim=-1)
        else:
            if not self.cfg['flags']['sequential']:
                logits_new = [
                    self.logits_norm(attn(embeddings, logits, hidden_states, supress_warnings=supress_warnings),
                                     supress_warnings=supress_warnings) for attn in self.attn]
                del embeddings, hidden_states
                logits_new = torch.stack(logits_new, dim=-1)
                if not self.cfg['flags']['layer_transform']:
                    logits_new = self.compress(logits_new).squeeze(-1)
                else:
                    logits_new = logits_new.sum(dim=-1, keepdim=False) / self.cfg['vals']['layers']
                logits += self.dropout(logits_new)
                del logits_new
            else:
                for i in range(self.cfg['vals']['layers']):
                    logits += self.dropout(
                        self.attn[i](embeddings, self.logits_norm(logits, supress_warnings=supress_warnings),
                                     hidden_states, supress_warnings=supress_warnings))
                logits = F.softmax(logits, dim=-1)
            return logits
