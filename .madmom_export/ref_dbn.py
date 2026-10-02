"""madmom DBN 参考实现（用于校验 C# 移植是否忠实）。

组成：
  - 真实 madmom 源码 `features/beats_hmm.py`（状态空间 / 转移模型 / 观测模型，纯 Python，直接 import）
  - `ml/hmm.pyx` 的 TransitionModel / ObservationModel / HiddenMarkovModel.viterbi 的纯 Python 逐行转写
    （make_sparse 用真实 scipy.csr_matrix，与 madmom 一致）
  - `features/downbeats.py` 的 DBNDownBeatTrackingProcessor.process 与
    `features/beats.py` 的 DBNBeatTrackingProcessor.process_offline 的逐行转写

用法：python ref_dbn.py <激活目录> [fps]
"""
import sys, os, types, importlib
import numpy as np

# madmom 0.16.1 用了 numpy 2.x 已移除的别名
np.float = float
np.int = int

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", ".madmom_pkg", "madmom-0.16.1", "madmom")
SRC = os.path.abspath(SRC)


def _pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


# ---------------------------------------------------------------- ml.hmm 转写
class TransitionModel(object):
    def __init__(self, states, pointers, probabilities):
        self.states = states
        self.pointers = pointers
        self.probabilities = probabilities

    @property
    def num_states(self):
        return len(self.pointers) - 1

    @property
    def log_probabilities(self):
        return np.log(self.probabilities)

    @staticmethod
    def make_sparse(states, prev_states, probabilities):
        from scipy.sparse import csr_matrix
        states = np.asarray(states)
        prev_states = np.asarray(prev_states, dtype=np.int)
        probabilities = np.asarray(probabilities)
        num_states = max(prev_states) + 1
        transitions = csr_matrix((probabilities, (states, prev_states)),
                                 shape=(num_states, num_states))
        states = transitions.indices.astype(np.uint32)
        pointers = transitions.indptr.astype(np.uint32)
        probabilities = transitions.data.astype(dtype=np.float)
        return states, pointers, probabilities


class ObservationModel(object):
    def __init__(self, pointers):
        self.pointers = pointers

    def log_densities(self, observations):
        raise NotImplementedError

    def densities(self, observations):
        return np.exp(self.log_densities(observations))


class HiddenMarkovModel(object):
    """ml/hmm.pyx viterbi() 的纯 Python 逐行转写。"""

    def __init__(self, transition_model, observation_model,
                 initial_distribution=None):
        self.transition_model = transition_model
        self.observation_model = observation_model
        if initial_distribution is None:
            initial_distribution = (np.ones(transition_model.num_states,
                                            dtype=np.float) /
                                    transition_model.num_states)
        self.initial_distribution = initial_distribution

    def viterbi(self, observations):
        tm = self.transition_model
        tm_states = tm.states
        tm_pointers = tm.pointers
        tm_probabilities = tm.log_probabilities
        num_states = tm.num_states
        om = self.observation_model
        num_observations = len(observations)
        om_pointers = om.pointers
        om_densities = om.log_densities(observations)

        current_viterbi = np.empty(num_states, dtype=np.float)
        previous_viterbi = np.log(self.initial_distribution)
        bt_pointers = np.empty((num_observations, num_states), dtype=np.uint32)

        for frame in range(num_observations):
            for state in range(num_states):
                current_viterbi[state] = -np.inf
                density = om_densities[frame, om_pointers[state]]
                for pointer in range(tm_pointers[state], tm_pointers[state + 1]):
                    prev_state = tm_states[pointer]
                    transition_prob = (previous_viterbi[prev_state] +
                                       tm_probabilities[pointer] + density)
                    if transition_prob > current_viterbi[state]:
                        current_viterbi[state] = transition_prob
                        bt_pointers[frame, state] = prev_state
            previous_viterbi = current_viterbi.copy()

        state = np.asarray(current_viterbi).argmax()
        log_probability = current_viterbi[state]
        if np.isinf(log_probability):
            return np.empty(0, dtype=np.uint32), log_probability
        path = np.empty(num_observations, dtype=np.uint32)
        for frame in range(num_observations - 1, -1, -1):
            path[frame] = state
            state = bt_pointers[frame, state]
        return path, log_probability


# -------- 注入 madmom 包结构，使真实 beats_hmm.py 可被 import --------
m = _pkg("madmom", os.path.dirname(SRC))
ml = _pkg("madmom.ml", os.path.join(SRC, "ml"))
hmm = types.ModuleType("madmom.ml.hmm")
hmm.TransitionModel = TransitionModel
hmm.ObservationModel = ObservationModel
hmm.HiddenMarkovModel = HiddenMarkovModel
sys.modules["madmom.ml.hmm"] = hmm
ml.hmm = hmm
feat = _pkg("madmom.features", os.path.join(SRC, "features"))
m.ml, m.features = ml, feat

beats_hmm = importlib.import_module("madmom.features.beats_hmm")


# ------------------------------------------- downbeats.py 的 processor 转写
def process_downbeat(activations, beats_per_bar, fps=100.0,
                     min_bpm=55.0, max_bpm=215.0, num_tempi=60,
                     transition_lambda=100, observation_lambda=16,
                     threshold=0.05, correct=True):
    min_interval = 60. * fps / max_bpm
    max_interval = 60. * fps / min_bpm
    hmms = []
    for beats in beats_per_bar:
        st = beats_hmm.BarStateSpace(beats, min_interval, max_interval, num_tempi)
        tm = beats_hmm.BarTransitionModel(st, transition_lambda)
        om = beats_hmm.RNNDownBeatTrackingObservationModel(st, observation_lambda)
        hmms.append(HiddenMarkovModel(tm, om))

    first = 0
    if threshold:
        idx = np.nonzero(activations >= threshold)[0]
        if idx.any():
            first = max(first, np.min(idx))
            last = min(len(activations), np.max(idx) + 1)
        else:
            last = first
        activations = activations[first:last]
    if not activations.any():
        return np.empty((0, 2))

    results = [h.viterbi(activations) for h in hmms]
    best = int(np.argmax(np.asarray([r[1] for r in results])))
    path, _ = results[best]
    st = hmms[best].transition_model.state_space
    om = hmms[best].observation_model
    positions = st.state_positions[path]
    beat_numbers = positions.astype(int) + 1

    beats = np.empty(0, dtype=np.int)
    if correct:
        beat_range = om.pointers[path] >= 1
        idx = np.nonzero(np.diff(beat_range.astype(np.int)))[0] + 1
        if beat_range[0]:
            idx = np.r_[0, idx]
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                peak = np.argmax(activations[left:right]) // 2 + left
                beats = np.hstack((beats, peak))
    else:
        beats = np.nonzero(np.diff(beat_numbers))[0] + 1
    return np.vstack(((beats + first) / float(fps), beat_numbers[beats])).T, beats_per_bar[best]


# ----------------------------------------------- beats.py 的 processor 转写
def process_beat(activations, fps=100.0, min_bpm=55.0, max_bpm=215.0,
                 num_tempi=None, transition_lambda=100, observation_lambda=16,
                 threshold=0, correct=True):
    min_interval = 60. * fps / max_bpm
    max_interval = 60. * fps / min_bpm
    st = beats_hmm.BeatStateSpace(min_interval, max_interval, num_tempi)
    tm = beats_hmm.BeatTransitionModel(st, transition_lambda)
    om = beats_hmm.RNNBeatTrackingObservationModel(st, observation_lambda)
    hmm_m = HiddenMarkovModel(tm, om, None)

    beats = np.empty(0, dtype=np.int)
    first = 0
    if threshold:
        idx = np.nonzero(activations >= threshold)[0]
        if idx.any():
            first = max(first, np.min(idx))
            last = min(len(activations), np.max(idx) + 1)
        else:
            last = first
        activations = activations[first:last]
    if not activations.any():
        return beats

    path, _ = hmm_m.viterbi(activations)
    if correct:
        beat_range = om.pointers[path]
        idx = np.nonzero(np.diff(beat_range))[0] + 1
        if beat_range[0]:
            idx = np.r_[0, idx]
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                peak = np.argmax(activations[left:right]) + left
                beats = np.hstack((beats, peak))
    else:
        from scipy.signal import argrelmin
        beats = argrelmin(st.state_positions[path], mode='wrap')[0]
        beats = beats[om.pointers[path[beats]] == 1]
    return (beats + first) / float(fps)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else ".madmom_export/act_dump"
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0

    db_act = np.loadtxt(os.path.join(d, "downbeat_act.txt"))
    b_act = np.loadtxt(os.path.join(d, "beat_act.txt"))
    print(f"downbeat act: {db_act.shape}   beat act: {b_act.shape}")

    # 状态空间规模（用于与 C# 对照）
    st3 = beats_hmm.BarStateSpace(3, 60. * fps / 215., 60. * fps / 55., 60)
    st4 = beats_hmm.BarStateSpace(4, 60. * fps / 215., 60. * fps / 55., 60)
    stb = beats_hmm.BeatStateSpace(60. * fps / 215., 60. * fps / 55., None)
    print(f"BarStateSpace(3): num_states={st3.num_states} "
          f"num_intervals={len(np.unique(st3.state_intervals))}")
    print(f"BarStateSpace(4): num_states={st4.num_states} "
          f"num_intervals={len(np.unique(st4.state_intervals))}")
    print(f"BeatStateSpace  : num_states={stb.num_states} num_intervals={stb.num_intervals}")

    res, bpb = process_downbeat(db_act, [3, 4], fps=fps)
    print(f"\n[REF] downbeat: {len(res)} 拍, 选中拍号={bpb}")
    for t, n in res:
        print(f"    t={t:7.3f}s  beat {int(n)}/{bpb}")

    bt = process_beat(b_act, fps=fps)
    print(f"\n[REF] beat: {len(bt)} 拍")
    for t in bt:
        print(f"    t={t:7.3f}s")


if __name__ == "__main__":
    main()
