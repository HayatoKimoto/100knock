import sys
import copy
from heapq import heappush, heappop
import torch

class BeamSearchNode(object):
    def __init__(self, tgt, prev_node, wid, logp, length):
        self.tgt = tgt
        self.prev_node = prev_node
        self.wid = wid
        self.logp = logp
        self.length = length

    def eval(self):
        return self.logp / float(self.length - 1 + 1e-6)


def beam_search_decoding(model,
                         src,
                         beam_width,
                         n_best,
                         sos_id,
                         eos_id,
                         pad_id,
                         max_dec_steps,
                         device):

    """Beam Seach Decoding for RNN
    Args:
        decoder: An RNN decoder model
        enc_outs: A sequence of encoded input. (T, bs, 2H). 2H for bidirectional
        enc_last_h: (bs, H)
        beam_width: Beam search width
        n_best: The number of output sequences for each input
    Returns:
        n_best_list: Decoded N-best results. (bs, T)
    """

    assert beam_width >= n_best

    # Decoding goes sentence by sentence.
    # So this process is very slow compared to batch decoding process.

    src = src.to(device)    # example: tensor([[   10,  3641,    14, 12805,   256,  6920,  2040,  6126,     8,   165]], device='cuda:1')
    src_mask = (src == pad_id).transpose(0,1).to(device)

    # Prepare first token for decoder
    tgt = torch.LongTensor([[sos_id]]).to(device)

    # Number of sentence to generate
    end_nodes = []

    # starting node
    node = BeamSearchNode(tgt=tgt, prev_node=None, wid=tgt.item(), logp=0, length=1)

    # whole beam search node graph
    nodes = []

    # Start the queue
    heappush(nodes, (-node.eval(), id(node), node))
    n_dec_steps = 0

    # Start beam search
    while True:
        # Give up when decoding takes too long
        if n_dec_steps > max_dec_steps:
            break

        # Fetch the best node
        score, _, n = heappop(nodes)
        tgt = n.tgt

        if tgt.size(0) >42 or( n.wid == eos_id and n.prev_node is not None):
            end_nodes.append((score, id(n), n))
            # If we reached maximum # of sentences required
            if len(end_nodes) >= n_best:
                break
            else:
                continue

        # Decode for one step using decoder
        pred = model(src.transpose(0,1), tgt, src_key_padding_mask=src_mask)
        pred = pred[-1].squeeze(0)

        # Get top-k from this decoded result
        topk_log_prob, topk_indexes = torch.topk(pred, beam_width) # (bw), (bw)
        # Then, register new top-k nodes
        for new_k in range(beam_width):
            decoded_t = topk_indexes[new_k].item() # int
            logp = topk_log_prob[new_k].item() # float log probability val
            tmp_tgt = torch.cat((tgt, torch.LongTensor([[decoded_t]]).to(device)))

            node = BeamSearchNode(tgt=tmp_tgt,
                                    prev_node=n,
                                    wid=decoded_t,
                                    logp=n.logp+logp,
                                    length=n.length+1)
            heappush(nodes, (-node.eval(), id(node), node))
        n_dec_steps += beam_width


    # if there are no end_nodes, retrieve best nodes (they are probably truncated)
    if len(end_nodes) == 0:
        end_nodes = [heappop(nodes) for _ in range(beam_width)]

    # Construct sequences from end_nodes
    n_best_seq_list = []
    for score, _id, n in sorted(end_nodes, key=lambda x: x[0])[:n_best]:
        sequence = [n.wid]
        # back trace from end node
        while n.prev_node is not None:
            n = n.prev_node
            sequence.append(n.wid)
        sequence = sequence[::-1] # reverse

        n_best_seq_list.append(sequence)


    return n_best_seq_list

