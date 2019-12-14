import os
import util
import numpy as np
from absl import flags
FLAGS = flags.FLAGS

def write_highlighted_html(html, out_dir, example_idx):
    html = '''

<button id="btnPrev" class="float-left submit-button" >Prev</button>
<button id="btnNext" class="float-left submit-button" >Next</button>
<br><br>

<script type="text/javascript">
    document.getElementById("btnPrev").onclick = function () {
        location.href = "%06d_highlighted.html";
    };
    document.getElementById("btnNext").onclick = function () {
        location.href = "%06d_highlighted.html";
    };

    document.addEventListener("keyup",function(e){
   var key = e.which||e.keyCode;
   switch(key){
      //left arrow
      case 37:
         document.getElementById("btnPrev").click();
      break;
      //right arrow
      case 39:
         document.getElementById("btnNext").click();
      break;
   }
});
</script>

''' % (example_idx - 1, example_idx + 1) + html
    path = os.path.join(out_dir, '%06d_highlighted.html' % example_idx)
    with open(path, 'w') as f:
        f.write(html)

highlight_colors = ['aqua', 'lime', 'yellow', '#FF7676', '#B9968D', '#D7BDE2', '#D6DBDF', '#F852AF', '#00FF8B', '#FD933A', '#8C8DFF', '#965DFF']
hard_highlight_colors = ['#00BBFF', '#00BB00', '#F4D03F', '#BB5454', '#A16252', '#AF7AC5', '#AEB6BF', '#FF008F', '#0ECA74', '#FF7400', '#6668FF', '#7931FF']

def start_tag(color):
    return "<font color='" + color + "'>"

def start_tag_highlight(color):
    return "<mark style='background-color: " + color + ";'>"

def get_idx_for_source_idx(similar_source_indices, source_idx):
    summ_sent_indices = []
    priorities = []
    for source_indices_idx, source_indices in enumerate(similar_source_indices):
        for idx_idx, idx in enumerate(source_indices):
            if source_idx == idx:
                summ_sent_indices.append(source_indices_idx)
                priorities.append(idx_idx)
    if len(summ_sent_indices) == 0:
        return None, None
    else:
        return summ_sent_indices, priorities

def html_highlight_sents_in_article(summary_sent_tokens, similar_source_indices_list,
                                    article_sent_tokens, doc_indices=None, lcs_paths_list=None, article_lcs_paths_list=None):
    end_tag = "</mark>"
    out_str = ''

    for summ_sent_idx, summ_sent in enumerate(summary_sent_tokens):
        try:
            similar_source_indices = similar_source_indices_list[summ_sent_idx]
        except:
            similar_source_indices = []

        for token_idx, token in enumerate(summ_sent):
            insert_string = token + ' '
            for source_indices_idx, source_indices in enumerate(similar_source_indices):
                if source_indices_idx == 0:
                    try:
                        color = hard_highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)]
                    except:
                        print(summ_sent_idx)
                        print(summary_sent_tokens)
                        print('\n')
                else:
                    color = highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)]
                if lcs_paths_list is None or token_idx in lcs_paths_list[summ_sent_idx][source_indices_idx]:
                    insert_string = start_tag_highlight(color) + token + ' ' + end_tag
                    break
            out_str += insert_string
        out_str += '<br><br>'

    cur_token_idx = 0
    cur_doc_idx = 0
    for sent_idx, sent in enumerate(article_sent_tokens):
        if doc_indices is not None:
            if cur_token_idx >= len(doc_indices):
                print("Warning: cur_token_idx is greater than len of doc_indices")
            elif doc_indices[cur_token_idx] != cur_doc_idx:
                cur_doc_idx = doc_indices[cur_token_idx]
                out_str += '<br>'
        summ_sent_indices, priorities = get_idx_for_source_idx(similar_source_indices_list, sent_idx)
        if priorities is None:
            colors = ['black']
            hard_colors = ['black']
        else:
            colors = [highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)] for summ_sent_idx in summ_sent_indices]
            hard_colors = [hard_highlight_colors[min(summ_sent_idx, len(highlight_colors)-1)] for summ_sent_idx in summ_sent_indices]
        source_sentence = article_sent_tokens[sent_idx]
        for token_idx, token in enumerate(source_sentence):
            if priorities is None:
                insert_string = token + ' '
            else:
                insert_string = token + ' '
                for priority_idx in reversed(list(range(len(priorities)))):
                    summ_sent_idx = summ_sent_indices[priority_idx]
                    priority = priorities[priority_idx]
                    if article_lcs_paths_list is None or token_idx in article_lcs_paths_list[summ_sent_idx][priority]:
                        if priority == 0:
                            insert_string = start_tag_highlight(hard_colors[priority_idx]) + token + ' ' + end_tag
                        else:
                            insert_string = start_tag_highlight(colors[priority_idx]) + token + ' ' + end_tag
            cur_token_idx += 1
            out_str += insert_string
        out_str += '<br>'
    out_str += '<br>------------------------------------------------------<br><br>'
    return out_str


def get_sent_similarities(summ_sent, article_sent_tokens, vocab):
    rouge_l = np.squeeze(util.rouge_l_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall'))
    rouge_1 = np.squeeze(util.rouge_1_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', True), 1)
    rouge_2 = np.squeeze(util.rouge_2_similarity_matrix(article_sent_tokens, [summ_sent], vocab, 'recall', False), 1)
    similarities = (rouge_l + rouge_1 + rouge_2) / 3.0
    return similarities

def get_simple_source_indices_list(summary_sent_tokens, article_sent_tokens, vocab=None, sentence_limit=2, min_matched_tokens=2):
    article_sent_tokens_lemma = util.lemmatize_sent_tokens(article_sent_tokens)
    summary_sent_tokens_lemma = util.lemmatize_sent_tokens(summary_sent_tokens)

    similar_source_indices_list = []
    lcs_paths_list = []
    smooth_article_paths_list = []
    for summ_sent in summary_sent_tokens_lemma:
        similarities = get_sent_similarities(summ_sent, article_sent_tokens_lemma, vocab)
        similar_source_indices, lcs_paths, smooth_article_paths = get_similar_source_sents_recursive(
            summ_sent, summ_sent, list(range(len(summ_sent))), article_sent_tokens_lemma, vocab, similarities, 0,
            sentence_limit, min_matched_tokens)
        similar_source_indices_list.append(similar_source_indices)
        lcs_paths_list.append(lcs_paths)
        smooth_article_paths_list.append(smooth_article_paths)
    deduplicated_similar_source_indices_list = []
    for sim_source_ind in similar_source_indices_list:
        dedup_sim_source_ind = []
        for ssi in sim_source_ind:
            if not (ssi in dedup_sim_source_ind or ssi[::-1] in dedup_sim_source_ind):
                dedup_sim_source_ind.append(ssi)
        deduplicated_similar_source_indices_list.append(dedup_sim_source_ind)
    simple_similar_source_indices = [tuple(sim_source_ind[0]) for sim_source_ind in deduplicated_similar_source_indices_list]
    lcs_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in lcs_paths_list]
    smooth_article_paths_list = [tuple(sim_source_ind[0]) for sim_source_ind in smooth_article_paths_list]
    return simple_similar_source_indices, lcs_paths_list, smooth_article_paths_list


# Recursive function
def get_similar_source_sents_recursive(summ_sent, partial_summ_sent, selection, article_sent_tokens, vocab, similarities, depth, sentence_limit, min_matched_tokens):
    if sentence_limit == 1:
        if depth > 2:
            return [[]], [[]], [[]]
    elif len(selection) < 3 or depth >= sentence_limit:      # base case: when summary sentence is too short
        return [[]], [[]], [[]]

    all_sent_indices = []
    all_lcs_paths = []
    all_smooth_article_paths = []

    # partial_summ_sent = util.reorder(summ_sent, selection)
    top_sent_indices, top_similarity = get_top_similar_sent(partial_summ_sent, article_sent_tokens, vocab)
    top_similarities = util.reorder(similarities, top_sent_indices)
    top_sent_indices = [x for _, x in sorted(zip(top_similarities, top_sent_indices), key=lambda pair: pair[0])][::-1]
    for top_sent_idx in top_sent_indices:
        nonstopword_matches, _ = util.matching_unigrams(partial_summ_sent, article_sent_tokens[top_sent_idx], should_remove_stop_words=True)
        lcs_len, (summ_lcs_path, _) = util.matching_unigrams(partial_summ_sent, article_sent_tokens[top_sent_idx])
        smooth_article_path = get_smooth_path(summ_sent, article_sent_tokens[top_sent_idx])
        if len(nonstopword_matches) < min_matched_tokens:
            continue
        leftover_selection = [idx for idx in range(len(partial_summ_sent)) if idx not in summ_lcs_path]
        partial_summ_sent = replace_with_blanks(partial_summ_sent, leftover_selection)

        sent_indices, lcs_paths, smooth_article_paths = get_similar_source_sents_recursive(
            summ_sent, partial_summ_sent, leftover_selection, article_sent_tokens, vocab, similarities, depth+1,
            sentence_limit, min_matched_tokens)   # recursive call

        combined_sent_indices = [[top_sent_idx] + indices for indices in sent_indices]      # append my result to the recursive collection
        combined_lcs_paths = [[summ_lcs_path] + paths for paths in lcs_paths]
        combined_smooth_article_paths = [[smooth_article_path] + paths for paths in smooth_article_paths]

        all_sent_indices.extend(combined_sent_indices)
        all_lcs_paths.extend(combined_lcs_paths)
        all_smooth_article_paths.extend(combined_smooth_article_paths)
    if len(all_sent_indices) == 0:
        return [[]], [[]], [[]]
    return all_sent_indices, all_lcs_paths, all_smooth_article_paths

def get_smooth_path(summ_sent, article_sent):
    summ_sent = ['<s>'] + summ_sent + ['</s>']
    article_sent = ['<s>'] + article_sent + ['</s>']

    matches = []
    article_indices = []
    summ_token_to_indices = util.create_token_to_indices(summ_sent)
    article_token_to_indices = util.create_token_to_indices(article_sent)
    for key in list(article_token_to_indices.keys()):
        if (util.is_punctuation(key) and not util.is_quotation_mark(key)):
            del article_token_to_indices[key]
    for token in list(summ_token_to_indices.keys()):
        if token in article_token_to_indices:
            article_indices.extend(article_token_to_indices[token])
            matches.extend([token] * len(summ_token_to_indices[token]))
    article_indices = sorted(article_indices)

    # Add a single word or a pair of words if they are in between two hightlighted content words
    new_article_indices = []
    new_article_indices.append(0)
    for article_idx in article_indices[1:]:
        word = article_sent[article_idx]
        prev_highlighted_word = article_sent[new_article_indices[-1]]
        if article_idx - new_article_indices[-1] <= 3 \
                and ((util.is_content_word(word) and util.is_content_word(prev_highlighted_word)) \
                or (len(new_article_indices) >= 2 and util.is_content_word(word) and util.is_content_word(article_sent[new_article_indices[-2]]))):
            in_between_indices = list(range(new_article_indices[-1] + 1, article_idx))
            are_not_punctuation = [not util.is_punctuation(article_sent[in_between_idx]) for in_between_idx in in_between_indices]
            if all(are_not_punctuation):
                new_article_indices.extend(in_between_indices)
        new_article_indices.append(article_idx)
    new_article_indices = new_article_indices[1:-1] # remove <s> and </s> from list

    # Remove isolated stopwords
    new_new_article_indices = []
    for idx, article_idx in enumerate(new_article_indices):
        if (not util.is_stopword_punctuation(article_sent[article_idx])) or (idx > 0 and new_article_indices[idx-1] == article_idx-1) or (idx < len(new_article_indices)-1 and new_article_indices[idx+1] == article_idx+1):
            new_new_article_indices.append(article_idx)
    new_new_article_indices = [idx-1 for idx in new_new_article_indices]    # fix indexing since we don't count <s> and </s>
    return new_new_article_indices

def get_top_similar_sent(summ_sent, article_sent_tokens, vocab):
    similarities = get_sent_similarities(summ_sent, article_sent_tokens, vocab)
    top_similarity = np.max(similarities)
    sent_indices = [np.argmax(similarities)]
    return sent_indices, top_similarity

def replace_with_blanks(summ_sent, selection):
    replaced_summ_sent = [summ_sent[token_idx] if token_idx in selection else '' for token_idx, token in enumerate(summ_sent)]
    return  replaced_summ_sent

def filter_pairs_by_sent_position(possible_pairs, rel_sent_indices=None):
    max_sent_position = {
        'cnn_dm': 30,
        'xsum': 20,
        'duc_2004': np.inf
    }
    if FLAGS.dataset_name == 'duc_2004':
        return [pair for pair in possible_pairs if max(rel_sent_indices[pair[0]], rel_sent_indices[pair[1]]) < 5]
    else:
        return [pair for pair in possible_pairs if max(pair) < max_sent_position[FLAGS.dataset_name]]

def get_rel_sent_indices(doc_indices, article_sent_tokens):
    if FLAGS.dataset_name != 'duc_2004' and len(doc_indices) != len(util.flatten_list_of_lists(article_sent_tokens)):
        doc_indices = [0] * len(util.flatten_list_of_lists(article_sent_tokens))
    doc_indices_sent_tokens = util.reshape_like(doc_indices, article_sent_tokens)
    if FLAGS.dataset_name != 'duc_2004':
        sent_doc = [0] * len(doc_indices_sent_tokens)
    else:
        sent_doc = [sent[0] for sent in doc_indices_sent_tokens]
    rel_sent_indices = []
    doc_sent_indices = []
    cur_doc_idx = 0
    rel_sent_idx = 0
    for doc_idx in sent_doc:
        if doc_idx != cur_doc_idx:
            rel_sent_idx = 0
            cur_doc_idx = doc_idx
        rel_sent_indices.append(rel_sent_idx)
        doc_sent_indices.append(cur_doc_idx)
        rel_sent_idx += 1
    doc_sent_lens = [sum(1 for my_doc_idx in doc_sent_indices if my_doc_idx == doc_idx) for doc_idx in
                     range(max(doc_sent_indices) + 1)]
    return rel_sent_indices, doc_sent_indices, doc_sent_lens