# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:59:50 2017

@author: Thomas Schatz

Utilities for matching corpora in terms of number of speakers, speaker genders,
and amount of speech material per speaker.

This is somewhat ad hoc code.

See also Xuan-nga scripts.
"""

import abkhazia.corpus
import io, shutil
import numpy as np


def sort_dict(d):
    """
    Sort keys and values in a dict based on values,
    in decreasing order   
    """
    keys = np.array(list(d.keys()))
    values = np.array([d[k] for k in keys])
    order = np.argsort(-values)  # sort opposites in increasing order
    res = [(keys[item], values[item]) for item in order]
    return res


def speaker_amount(corpus):
    """
    Get total duration for each speaker in the corpus. Only parts
    within a valid annotated utterance are counted. Noise, silence
    and otherwise untranscribed parts are ignored.
    """
    durations = corpus.utt2duration()
    spk2dur = {}
    for utt in durations:
        spk = corpus.utt2spk[utt]
        dur = durations[utt]
        if spk in spk2dur:
            spk2dur[spk] = spk2dur[spk] + dur
        else:
            spk2dur[spk] = dur
    return spk2dur


def parse_gender_file(gender_file):
    """
    Returns a dict: (spk, gender).
    """
    spk2gender = {}
    with io.open(gender_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            spk, gender = line.strip().split(' ')
            assert gender in ['m', 'f'], ("Unsupported gender {} for speaker "
                                          "{} gender should be m or f").format(
                                          gender, spk)
            spk2gender[spk] = gender
    return spk2gender
    
    
def spk_amount_by_gender(spk2dur, spk2gender):
    """
    Return amount by speaker separately for each gender,
    sorted in decreasing order of duration.
    """
    # some consistency checks
    spk1 = set(spk2dur.keys())
    spk2 = set(spk2gender.keys())
    diff1 = spk1.difference(spk2)
    diff2 = spk2.difference(spk1)
    assert diff1 == set(), "Gender is missing for speakers: {}".format(diff1)
    if not(diff2 == set()):
        print("Gender given for speakers not in the corpus: {}".format(diff2))
    
    # split in males and females
    male2dur = {spk: spk2dur[spk]
                    for spk in spk2dur if spk2gender[spk] == 'm'}
    female2dur = {spk: spk2dur[spk]
                    for spk in spk2dur if spk2gender[spk] == 'f'}
    return {'females': sort_dict(female2dur), 'males': sort_dict(male2dur)}


def load_spk_matching_data(corpora_folders, gender_files):
    """
    Load data necessary to perform corpus matching, i.e.
    for each corpora the amount of speech material available
    by speaker, separately for each gender.
    """
    # copy gender files into corpus data folders
    for corpus in corpora_folders:
        shutil.copy(gender_files[corpus], corpora_folders[corpus])
    # load data
    spk_amounts = {}
    for corpus_name in corpora_folders:
        spk2gender = parse_gender_file(gender_files[corpus_name])
        corpus = abkhazia.corpus.Corpus.load(corpora_folders[corpus_name],
                                             validate=True)
        spk2dur = speaker_amount(corpus)
        spk_amounts[corpus_name] = spk_amount_by_gender(spk2dur, spk2gender)
    return spk_amounts


def match_speakers_big2small(matching_data, big_corpus, small_corpus,
                             threshold=None):
    """
    Match a larger corpus to a small corpus: for each speaker in the small
    corpus, associate the speaker from the big corpus with the same gender and
    the closest amount of speech, return the association, the relative
    difference of duration between the associated speakers: 
    2*(dur_big_corpus - dur_small_corpus) / (dur_big_corpus + dur_small_corpus)
    and the total duration for each corpus and gender.
    
    If threshold is not None, speaker from the small corpus are allowed
    to be paired with speakers for which the relative difference in duration is
    larger than threshold only if they are the shortest of the pair.
    """
    genders = ['females', 'males']
    spk_match = {small_corpus : [], big_corpus : []}
    spk_gender = []
    relative_diff = []
    gender_dur = {}
    gender_dur[small_corpus] = {gender : 0. for gender in genders}
    gender_dur[big_corpus] = {gender : 0. for gender in genders}
    for gender in genders:
        data = matching_data[big_corpus][gender]
        big_durs = np.array([d for (s, d) in data])
        big_spks = np.array([s for (s, d) in data])
        for spk, dur in matching_data[small_corpus][gender]:
            # find and register match
            diff = np.abs(big_durs - dur)
            ind = np.argmin(diff)
            rel_d = 2*(big_durs[ind] - dur)/(big_durs[ind] + dur)
            if not(threshold is None) and abs(rel_d) > threshold:
                # find larger file
                diff = big_durs - dur
                diff[np.where(diff < 0)[0]] = np.inf
                assert not(all(diff == np.inf)), \
                    ("Speaker {} from {} cannot "
                     "be matched in {}").format(spk, small_corpus, big_corpus)
                ind = np.argmin(diff)
                rel_d = 2*(big_durs[ind] - dur)/(big_durs[ind] + dur)
            relative_diff.append(rel_d)
            spk_match[small_corpus].append(spk)
            spk_gender.append(gender)
            gender_dur[small_corpus][gender] += dur
            spk_match[big_corpus].append(big_spks[ind])
            gender_dur[big_corpus][gender] += big_durs[ind]
            # remove matched speaker from speakers available for matching
            big_durs = np.concatenate([big_durs[:ind], big_durs[ind+1:]])
            big_spks = np.concatenate([big_spks[:ind], big_spks[ind+1:]])
            assert len(big_spks) > 0, \
                "Not enough speakers in corpus {}".format(big_corpus)
    return spk_match, spk_gender, relative_diff, gender_dur


def trim_utts(corpus_name, corpus, spk, target_dur, current_dur):
    """
    Find utterances to be removed so that the amount of speech
    in corpus 'corpus' by speaker 'spk' goes from 'current_dur' to 'target_dur' 
    seconds (approximately).
    Utterances are removed one by one, starting from the end of a recording,
    until the amount target_dur is reached. If there are several recordings
    from the desired speaker, we start removing from the shortest one, then
    the next shortest, etc.
    """
    print(("Trimming corpus {} speaker {} "
           "from {} to {} seconds").format(corpus_name, spk,
                                           current_dur, target_dur))
    assert target_dur <= current_dur
    spk_utts = [utt for utt in corpus.utt2spk if corpus.utt2spk[utt] == spk]
    # get speaker utterances by recording    
    spk_segs = {}
    for utt in spk_utts:
        seg = corpus.segments[utt][0]
        if seg in spk_segs:
            spk_segs[seg].append(utt) 
        else:
            spk_segs[seg] = [utt]
    # get utterances in reverse order of occurrence (last to first)
    spk_segs = {seg: np.array(spk_segs[seg]) for seg in spk_segs}
    for seg in spk_segs:
        if len(spk_segs[seg]) > 1:
            utt_starts = np.array([corpus.segments[utt][1]
                                    for utt in spk_segs[seg]])
            utt_order = np.argsort(utt_starts)[::-1]  # decreasing order
            spk_segs[seg] = spk_segs[seg][utt_order]
    # get recordings in increasing order of duration
    durations = corpus.utt2duration()
    seg_ids = np.array([seg for seg in spk_segs])   
    seg_durs = np.array([sum([durations[utt] for utt in spk_segs[seg]])
                          for seg in seg_ids])
    order = np.argsort(seg_durs)
    sorted_seg_ids = seg_ids[order]
    # get utt_ids to exclude
    spurious_utts = []
    assert abs(current_dur - sum(seg_durs)) < 1e-8  # handle rounding errors
    stop = False
    for seg in sorted_seg_ids:
        if stop:
            break
        for utt in spk_segs[seg]:
            dur = durations[utt]
            if current_dur - dur > target_dur:
                spurious_utts.append(utt)
                current_dur -= durations[utt]
            else:
                stop = True
                # decide what to do with current utt
                err1 = abs(target_dur - current_dur) 
                err2 = abs(target_dur - current_dur + dur)
                if err1 > err2:
                    spurious_utts.append(utt)  # remove current uttt
                    current_dur -= durations[utt]
                break               
    return spurious_utts


def create_matching_corpora(in_corpora, out_corpora, spk_match,
                            trim=False, relative_diff=None, threshold=.05):
    """    
    Create subcorpora containing only the speakers
    appearing in the match between two or more corpora.
    
    The trim option is currently only supported if there are only two corpora.
    If trim=true, matched speaker pairs for which the relative difference in
    available duration is larger than the provided 'threshold' are balanced by
    removing some utterances from the speaker with most material (see trim_utts
    for details).
    Note that the particular process used by trim_utts might not be appropriate
    if recordings in the corpus are done utterance by utterance, because it
    will remove the shorter utterances first independent of a possible natural
    order between occurrences in the corpus.
    Also note that this does not affect wavefiles at all, only
    annotations are removed.  We'd need to be careful of cases with several
    speakers by recording if we'd want to affect wavefiles...
    """
    # load corpora
    corpora = {}
    for corpus in in_corpora:
        print("Loading corpus {}".format(corpus))
        corpus_path = in_corpora[corpus]
        corpora[corpus] = abkhazia.corpus.Corpus.load(corpus_path,
                                                      validate=True)
    # find list of utt_ids to keep for each corpus
    utt_ids = {}                                
    for corpus_name in corpora:
        corpus = corpora[corpus_name]
        speakers = spk_match[corpus_name]
        # utt from desired speakers
        utt_ids[corpus_name] = {utt_id
                                    for utt_id in corpus.utt2spk
                                        if corpus.utt2spk[utt_id] in speakers}
        print(("Selecting {} utterances out "
               "of {} for corpus {}").format(len(utt_ids[corpus_name]),
                                             len(corpus.utt2spk), corpus_name))
    if trim:
        assert len(corpora) == 2
        assert not(relative_diff is None)
        spk2dur = {c: speaker_amount(corpora[c]) for c in corpora}
        for i in range(len(relative_diff)):
            if abs(relative_diff[i]) > threshold:
                c_names = list(corpora.keys())
                durs = [spk2dur[c][spk_match[c][i]] for c in c_names]
                target_c = c_names[0] if durs[0] > durs[1] else c_names[1]
                spk = spk_match[target_c][i]
                target_dur = min(durs)
                current_dur = max(durs)
                spurious_utts = trim_utts(target_c, corpora[target_c], spk,
                                          target_dur, current_dur)
                utt_ids[target_c] = utt_ids[target_c].difference(spurious_utts)   
    # instantiate subcorpora
    for corpus_name in corpora:
        print("Saving subcorpora for {}".format(corpus_name))
        corpus = corpora[corpus_name]
        subcorpus = corpus.subcorpus(list(utt_ids[corpus_name]), prune=True,
                                     name=corpus.meta.name+'-balanced',
                                     validate=True)    
        subcorpus.save(out_corpora[corpus_name], copy_wavs=False)
        #TODO? filter and copy alignment file too ?
        #shutil.copy(alignment_file, output_folder)
    # maybe log some stuff too ? like list of removed speakers and amount 
    # trimmed of kept speakers where applicable
  

def speakers_subcorpus(corpus_name, in_path, out_path, speakers):
    # load corpus
    print("Creating corpus {}".format(corpus_name))
    corpus = abkhazia.corpus.Corpus.load(in_path, validate=True)
    # find list of utt_ids to keep
    utt_ids = {utt_id for utt_id in corpus.utt2spk
                        if corpus.utt2spk[utt_id] in speakers}
    print(("Selecting {} utterances out "
           "of {}").format(len(utt_ids), len(corpus.utt2spk)))
    # instantiate subcorpora
    subcorpus = corpus.subcorpus(list(utt_ids), prune=True,
                                     name=corpus.meta.name+'-'+corpus_name,
                                     validate=True)    
    subcorpus.save(out_path, copy_wavs=False)


"""
def twin_balanced_split(spk_match, spk_gender, in_path, out_path, seed=12345):
    Split two matched corpora in equal-size train-test subcorpora, balanced in
    terms of speaker genders, number of speakers and amount by speaker
    distributions
    
    spk_match give the matched speakers in decreasing order of duration but
    grouped by gender
    
    Take two consecutive pairs of matched speakers from the two corpora,
    assign one pair to the train set the other to the test set at random.
    np.random.seed(seed)
    corpora = [corpus for corpus in spk_match]
    genders = ['males', 'females'] 
    train_spk, test_spk = {c : [] for c in corpora}, {c : [] for c in corpora}
    for gender in genders:
        speakers = {c: [spk for spk, g in zip(spk_match[c], spk_gender)
                                if g == gender] 
                    for c in spk_match}
        nb_speakers = len([g for g in spk_gender if g == gender])
        if nb_speakers % 2 == 1:
            # if the number of speakers is odd, the shortest speaker is
            # added to the train set
            for c in corpora:
                train_spk[c].append(speakers[c][-1])
            nb_speakers = nb_speakers - 1
        nb_flips = nb_speakers // 2
        flips = np.random.randint(2, size=nb_flips)
        train_ind = [2*i+j for (i,j) in zip(range(nb_flips), flips)]
        test_ind = [2*i+1-j for (i,j) in zip(range(nb_flips), flips)]
        for corpus in corpora:
            train_spk[corpus] = train_spk[corpus] + \
                                [speakers[corpus][i] for i in train_ind]
            test_spk[corpus] = test_spk[corpus] + \
                                [speakers[corpus][i] for i in test_ind]
    #instantiate subcorpora
    for corpus in corpora:
        speakers_subcorpus('train',
                           in_path[corpus], out_path[corpus] + '_train',
                           train_spk[corpus])
        speakers_subcorpus('test',
                           in_path[corpus], out_path[corpus] + '_test',
                           test_spk[corpus])
"""


def twin_balanced_split(spk_match, spk_gender, in_path, out_path,
                        seed=12345, group_size=2, train_proportion=1):
    """
    Split two matched corpora in train-test subcorpora, balanced in
    terms of speaker genders, number of speakers and amount by speaker
    distributions in a train_proportion/group_size ratio
    
    spk_match give the matched speakers in decreasing order of duration but
    grouped by gender
    
    Take 'group_size' consecutive pairs of matched speakers 
    from the two corpora; assign 'train_proportion' pairs to
    the train set and the others to the test set at random.

    The match between train and test is loose with this method
    and more importance is given to the match between the test
    of the two corpus and the train of the two corpus.
    """
    assert train_proportion <= group_size
    ratio = train_proportion/float(group_size)
    np.random.seed(seed)
    corpora = [corpus for corpus in spk_match]
    genders = ['males', 'females'] 
    train_spk, test_spk = {c : [] for c in corpora}, {c : [] for c in corpora}
    for gender in genders:
        speakers = {c: [spk for spk, g in zip(spk_match[c], spk_gender)
                                if g == gender] 
                    for c in spk_match}
        nb_speakers = len([g for g in spk_gender if g == gender])
        r = nb_speakers % group_size
        ## Edge case
        # if the number of speakers is not a multiple of group_size, 
        # let us note r the remainder
        # n(r) of the r shortest speakers are added
        # to the train set, the others to the test set
        # where n(r) is the minimum integer such that
        # n(r)/r >= train_proportion/group_size
        if r > 0:
            n_train = [e for e in np.arange(r+1) if e/float(r)>=ratio][0]
        else:
            n_train = 0
        arr = np.arange(r)
        np.random.shuffle(arr)
        train_ind, test_ind = arr[:n_train], arr[n_train:]
        train_ind = nb_speakers - 1 - train_ind
        test_ind = nb_speakers - 1 - test_ind
        nb_speakers = nb_speakers - r
        ## Main case
        nb_flips = nb_speakers // group_size
        for i in range(nb_flips):
            offset = group_size*i
            arr = np.arange(group_size)
            np.random.shuffle(arr)
            tr_ind, te_ind = arr[:train_proportion], arr[train_proportion:]
            tr_ind = tr_ind + offset
            te_ind = te_ind + offset
            train_ind = np.concatenate([train_ind, tr_ind])
            test_ind = np.concatenate([test_ind, te_ind])
        for corpus in corpora:
            train_spk[corpus] = train_spk[corpus] + \
                                [speakers[corpus][i] for i in train_ind]
            test_spk[corpus] = test_spk[corpus] + \
                                [speakers[corpus][i] for i in test_ind]
    #instantiate subcorpora
    for corpus in corpora:
        speakers_subcorpus('train',
                           in_path[corpus], out_path[corpus] + '_train',
                           train_spk[corpus])
        speakers_subcorpus('test',
                           in_path[corpus], out_path[corpus] + '_test',
                           test_spk[corpus])
