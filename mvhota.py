import math
import numpy as np
from scipy.optimize import linear_sum_assignment


class GetMvHotaSequence:
    def __init__(self, gt_dets_seq_left, gt_dets_seq_right, tracker_dets_seq_left, tracker_dets_seq_right):
        self.tracker_dets_seq_left = tracker_dets_seq_left
        self.tracker_dets_seq_right = tracker_dets_seq_right

        self.global_labels = GetMvHotaSequence.get_union_of_n_lists([list(gt_dets_seq_left.keys()),
                                                                list(gt_dets_seq_right.keys())])

        self.global_to_left = {left_id: i for i, left_id in enumerate(gt_dets_seq_left)}
        self.global_to_right = {right_id: i for i, right_id in enumerate(gt_dets_seq_right)}

        gt_dets_seq_left_local = []
        for frame in gt_dets_seq_left:
            gt_dets_frame_local = {}
            for id_, pt in frame:
                gt_dets_frame_local[self.global_to_left[id_]] = pt
            gt_dets_seq_left_local.append(gt_dets_frame_local)

        gt_dets_seq_right_local = []
        for frame in gt_dets_seq_right:
            gt_dets_frame_local = {}
            for id_, pt in frame:
                gt_dets_frame_local[self.global_to_right[id_]] = pt
            gt_dets_seq_right_local.append(gt_dets_frame_local)

        self.gt_dets_seq_left = gt_dets_seq_left_local
        self.gt_dets_seq_right = gt_dets_seq_right_local

        self.left_to_global = {v: k for k, v in self.global_to_left.items()}
        self.right_to_global = {v: k for k, v in self.global_to_right.items()}

    @staticmethod
    def get_distance_point(gt_point, tracker_point):
        return abs(math.sqrt((gt_point[0] - tracker_point[0]) ** 2 + (gt_point[1] - tracker_point[1]) ** 2))

    @staticmethod
    def get_distance_frame(gt_points_frame, tracker_points_frame, radius=6, max_distance=588):
        distance = np.zeros((len(gt_points_frame), len(tracker_points_frame)))
        for i, gt_point in enumerate(gt_points_frame):
            for j, tracker_point in enumerate(tracker_points_frame):
                norm = GetMvHotaSequence.get_distance_point(gt_point=gt_point,
                                                 tracker_point=tracker_point)
                distance[i][j] = norm if norm <= radius else max_distance + 1
        return distance

    @staticmethod
    def get_union_of_n_lists(list_of_lists):
        list_of_sets = [set(list_elem) for list_elem in list_of_lists]
        return list(set.union(*list_of_sets))

    def get_mvhota(self):
        left_matched_regions, left_metric = self.matching(self.gt_dets_seq_left, self.tracker_dets_seq_left)
        left_matched_pred_dicts = GetMvHotaSequence.get_matched_pred_dicts(left_matched_regions,
                                                              self.gt_dets_seq_left, self.tracker_dets_seq_left)

        right_matched_regions, right_metric = self.matching(self.gt_dets_seq_right, self.tracker_dets_seq_right)
        right_matched_pred_dicts = GetMvHotaSequence.get_matched_pred_dicts(right_matched_regions,
                                                               self.gt_dets_seq_right, self.tracker_dets_seq_right)

        detacc = 0.5 * (left_metric['detacc'] + right_metric['detacc'])
        tempassc = 0.5 * (left_metric['tempassc'] + right_metric['tempassc'])
        mvassc = self.get_mv_assc(left_matched_pred_dicts, right_matched_pred_dicts,
                                  self.tracker_dets_seq_left, self.tracker_dets_seq_right)
        mvhota = (detacc * tempassc * mvassc) ** (1/3)
        return mvhota

    def matching(self, gt_dets_seq, tracker_dets_seq):
        """
        Function to perform matching using the hungarian algorithm, given GT detections and Tracker detections
        This code snippet is modified from the original here: https://github.com/JonathonLuiten/TrackEval
        """
        [gt_ids, tracker_ids] = map(lambda x: [list(xi.keys()) for xi in x], [gt_dets_seq, tracker_dets_seq])

        num_gt_ids = len(list(set().union(*gt_ids)))
        num_tracker_ids = len(list(set().union(*tracker_ids)))

        potential_matches_count = np.zeros((num_gt_ids, num_tracker_ids))
        gt_id_count = np.zeros((num_gt_ids, 1))
        tracker_id_count = np.zeros((1, num_tracker_ids))

        matched_gt_dets = [[]]*len(gt_ids)  # Length of sequence
        matched_tracker_dets = [[]]*len(tracker_ids)  # Length of sequence
        num_matches = 0

        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
            gt_ids_t = np.asarray(gt_ids_t)
            tracker_ids_t = np.asarray(tracker_ids_t)

            gt_points_t = [gt_dets_seq[t][index] for index in gt_ids_t]
            tracker_points_t = [tracker_dets_seq[t][index] for index in tracker_ids_t]

            similarity = MvHOTA.get_distance_frame(gt_points_frame=gt_points_t,
                                                   tracker_points_frame=tracker_points_t)
            similarity[similarity > 6] = 588
            sim_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim = np.zeros_like(similarity)
            sim_mask = sim_denom > 0 + np.finfo('float').eps
            sim[sim_mask] = similarity[sim_mask] / sim_denom[sim_mask]

            if len(gt_ids_t) and len(tracker_ids_t):
                potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim
            if len(gt_ids_t):
                gt_id_count[np.asarray(gt_ids_t)] += 1
            if len(tracker_ids_t):
                tracker_id_count[0, np.asarray(tracker_ids_t)] += 1

        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_count = np.zeros_like(potential_matches_count)

        metric = {'TP': 0,
                  'FN': 0,
                  'FP': 0}

        matches = []
        ####################################### (4)
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
            gt_ids_t = np.asarray(gt_ids_t)
            tracker_ids_t = np.asarray(tracker_ids_t)

            gt_points_t = [gt_dets_seq[t][index] for index in gt_ids_t]
            tracker_points_t = [tracker_dets_seq[t][index] for index in tracker_ids_t]

            similarity = MvHOTA.get_distance_frame(gt_points_frame=gt_points_t,
                                                   tracker_points_frame=tracker_points_t)
            similarity[similarity > 6] = 588

            if len(gt_ids_t) and len(tracker_ids_t):
                score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
                match_rows, match_cols = linear_sum_assignment(score_mat)

                actually_matched_mask = similarity[match_rows, match_cols] <= 6
                match_rows_th6 = match_rows[actually_matched_mask]
                match_cols_th6 = match_cols[actually_matched_mask]
                num_matches = len(match_rows_th6)

                matched_gt_dets[t] = [gt_dets_seq[t][id_] for id_ in gt_ids_t[match_rows_th6]]
                matched_tracker_dets[t] = [tracker_dets_seq[t][id_] for id_ in tracker_ids_t[match_cols_th6]]

            metric['TP'] += num_matches
            metric['FN'] += len(gt_ids_t) - num_matches
            metric['FP'] += len(tracker_ids_t) - num_matches

        matched_regions = [[(tracker_point, gt_point) for tracker_point, gt_point in zip(tracker_det, gt_det)]
                           for tracker_det, gt_det in zip(matched_tracker_dets, matched_gt_dets)]

        ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
        metric['tempassc'] = np.sum(matches_count * ass_a) / np.maximum(1, metric['TP'])
        metric['detacc'] = metric['TP'] / np.maximum(1, metric['TP'] + metric['FN'] + metric['FP'])
        return matched_regions, metric

    @staticmethod
    def get_matched_pred_dicts(matched_regions_seq, gt_frame_dicts_seq, tracker_frame_dets_seq):
        tp_gt_pts_gt_ids_seq = []
        tp_pred_pts_gt_ids_seq, tp_pred_pts_pred_ids_seq = [], []
        fn_gt_pts_gt_ids_seq, fp_pred_pts_pred_ids_seq = [], []
        tp_seq, fp_seq, fn_seq = [], [], []

        for gt_frame, pred_frame, matched_regions_frame in zip(gt_frame_dicts_seq,
                                                               tracker_frame_dets_seq,
                                                               matched_regions_seq):
            tp_gt_pts_gt_ids_frame = {}
            tp_pred_pts_gt_ids_frame, tp_pred_pts_pred_ids_frame = {}, {}
            fn_gt_pts_gt_ids_frame, fp_pred_pts_pred_ids_frame = {}, {}

            for i, (pred_match, gt_match) in enumerate(matched_regions_frame):
                matched_gt_label = list(gt_frame.keys())[list(gt_frame.values()).index(gt_match)]
                matched_pred_label = list(pred_frame.keys())[list(pred_frame.values()).index(pred_match)]

                tp_gt_pts_gt_ids_frame[matched_gt_label] = gt_match
                tp_pred_pts_gt_ids_frame[matched_gt_label] = pred_match
                tp_pred_pts_pred_ids_frame[matched_pred_label] = pred_match

                fn_gt_pts_gt_ids_frame[matched_gt_label] = ()
                fp_pred_pts_pred_ids_frame[matched_pred_label] = ()

            for label, point in gt_frame.items():
                if label not in tp_gt_pts_gt_ids_frame.keys():
                    tp_gt_pts_gt_ids_frame[label] = ()
                    fn_gt_pts_gt_ids_frame[label] = point

            for label, point in pred_frame.items():
                if label not in tp_pred_pts_pred_ids_frame:
                    fp_pred_pts_pred_ids_frame[label] = point
                    tp_pred_pts_pred_ids_frame[label] = ()

            tp_seq.append(len(matched_regions_frame))  # number of pred_matches
            fp_seq.append(len(list(pred_frame.keys())) - len(matched_regions_frame))
            fn_seq.append(len(list(gt_frame.keys())) - len(matched_regions_frame))

            tp_gt_pts_gt_ids_seq.append(tp_gt_pts_gt_ids_frame)
            tp_pred_pts_gt_ids_seq.append(tp_pred_pts_gt_ids_frame)
            tp_pred_pts_pred_ids_seq.append(tp_pred_pts_pred_ids_frame)
            fn_gt_pts_gt_ids_seq.append(fn_gt_pts_gt_ids_frame)
            fp_pred_pts_pred_ids_seq.append(fp_pred_pts_pred_ids_frame)

        return {'tp_gt_pts_gt_ids_seq': tp_gt_pts_gt_ids_seq,
                'tp_pred_pts_gt_ids_seq': tp_pred_pts_gt_ids_seq,
                'tp_pred_pts_pred_ids_seq': tp_pred_pts_pred_ids_seq,
                'fn_gt_pts_gt_ids_seq': fn_gt_pts_gt_ids_seq,
                'fp_pred_pts_pred_ids_seq': fp_pred_pts_pred_ids_seq,
                'tp_seq': tp_seq, 'fp_seq': fp_seq, 'fn_seq': fn_seq}

    def get_mv_assc(self, left_matched_dicts, right_matched_dicts,
                            left_tracker_dets_seq, right_tracker_dets_seq):

        tpc = 0
        fpc = 0
        fnc = 0

        for frame in range(len(left_matched_dicts['tp_pred_pts_gt_ids_seq'])):
            for left_gt_id, left_pred_point in left_matched_dicts['tp_pred_pts_gt_ids_seq'][frame]:
                    tpc += 1
                    left_pred_id = list(left_matched_dicts['tp_pred_pts_pred_ids_seq'][frame].keys())[
                        list(left_matched_dicts['tp_pred_pts_pred_ids_seq'][frame].values()).index(left_pred_point)]

                    if self.left_to_global[left_gt_id] in list(self.global_to_right.keys()):
                        right_gt_id = self.global_to_right[self.left_to_global[left_gt_id]]

                        if right_gt_id in list(right_matched_dicts['tp_gt_pts_gt_ids_seq'].keys()):
                            right_pred_pt = right_matched_dicts['tp_pred_pts_gt_ids_seq'][frame][right_gt_id]
                            right_pred_id = list(right_matched_dicts['tp_pred_pts_pred_ids_seq'][frame].keys())[
                        list(left_matched_dicts['tp_pred_pts_pred_ids_seq'][frame].values()).index(right_pred_pt)]

                            if left_pred_id == right_pred_id: tpc += 1
                            else: fnc += 1

                    if left_pred_id in list(right_matched_dicts['tp_pred_pts_pred_ids_seq'].keys()):
                        right_pred_pt = right_matched_dicts['tp_pred_pts_pred_ids_seq'][left_pred_id]
                        right_gt_id = list(right_matched_dicts['tp_pred_pts_gt_ids_seq'].keys())[
                            list(right_matched_dicts['tp_pred_pts_gt_ids_seq'].values()).index(right_pred_pt)]
                        if left_gt_id != right_gt_id:
                            fpc += 1
                    elif left_pred_id in list(right_tracker_dets_seq[frame].keys()):
                        fpc += 1

            for right_gt_id, right_pred_point in right_matched_dicts['tp_pred_pts_gt_ids_seq'][frame]:
                tpc += 1
                right_pred_id = list(right_matched_dicts['tp_pred_pts_pred_ids_seq'][frame].keys())[
                    list(right_matched_dicts['tp_pred_pts_pred_ids_seq'][frame].values()).index(right_pred_point)]

                if self.right_to_global[right_gt_id] in list(self.global_to_left.keys()):
                    left_gt_id = self.global_to_left[self.right_to_global[right_gt_id]]

                    if left_gt_id in list(left_matched_dicts['tp_gt_pts_gt_ids_seq'].keys()):
                        left_pred_pt = left_matched_dicts['tp_pred_pts_gt_ids_seq'][frame][left_gt_id]
                        left_pred_id = list(left_matched_dicts['tp_pred_pts_pred_ids_seq'][frame].keys())[
                            list(right_matched_dicts['tp_pred_pts_pred_ids_seq'][frame].values()).index(left_pred_pt)]

                        if right_pred_id == left_pred_id: tpc += 1
                        else: fnc += 1

                if right_pred_id in list(left_matched_dicts['tp_pred_pts_pred_ids_seq'].keys()):
                    left_pred_pt = left_matched_dicts['tp_pred_pts_pred_ids_seq'][right_pred_id]
                    left_gt_id = list(left_matched_dicts['tp_pred_pts_gt_ids_seq'].keys())[
                        list(left_matched_dicts['tp_pred_pts_gt_ids_seq'].values()).index(left_pred_pt)]
                    if right_gt_id != left_gt_id:
                        fpc += 1
                elif right_pred_id in list(left_tracker_dets_seq[frame].keys()):
                    fpc += 1

        mv_assc = tpc / ((tpc + fpc + fnc) + np.finfo(float).eps)
        return mv_assc






