# -*- coding: utf-8 -*-
from local.utils import ProgressBar
from tools.data_helper import load_js
from config import *
import pandas as pd
from time import time

import jieba
import jieba.analyse

import sys, os, collections, copy
from recall.ann_search import FaissSearch

sys.path.append('./keyword_bert/')
from keyword_bert import tokenization
# from keyword_bert import tokenization, modeling, optimization
import tensorflow as tf
import re, time

import json
import requests


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 kw_mask_a_raw=None, kw_mask_b_raw=None):
        """Constructs a InputExample.
    
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          kw_mask_a_raw: list, [0, 1, 1] means 2nd and 3rd token in text_a is keyword
          kw_mask_b_raw: same as kw_mask_a_raw except on text_b
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.kw_mask_a_raw = kw_mask_a_raw
        self.kw_mask_b_raw = kw_mask_b_raw
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
  
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True,
                 kw_mask_a=None,
                 kw_mask_b=None,
                 real_mask_a=None,
                 real_mask_b=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        if kw_mask_a and kw_mask_b:
            self.kw_mask_a = kw_mask_a
            self.kw_mask_b = kw_mask_b
        if real_mask_a and real_mask_b:
            self.real_mask_a = real_mask_a
            self.real_mask_b = real_mask_b
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            # for line in reader:
            for line in f:
                line = line.strip().split('\t')
                lines.append(line)
            return lines


class BaikeProcessor(DataProcessor):
    """Process baike data with keyword (q-q/q-a pairs)
  
      File line format:
        `label \t text_a \t keyword_a_mask' \t text_b \t keyword_b_mask`
        example: `1 \t 我 爱 你 中 国 \t 0 1 0 1 1 \t 爱 我 中 华 \t 1 0 1 1`
    """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_predict_examples(self, file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(file), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #   continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            kw_mask_a_raw = list(map(int, line[2].split(' ')))
            text_b = tokenization.convert_to_unicode(line[3])
            kw_mask_b_raw = list(map(int, line[4].split(' ')))
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                             kw_mask_a_raw=kw_mask_a_raw, kw_mask_b_raw=kw_mask_b_raw))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`.
        example contains text keywords
    """

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            kw_mask_a=[0] * max_seq_length,
            kw_mask_b=[0] * max_seq_length,
            real_mask_a=[0] * max_seq_length,
            real_mask_b=[0] * max_seq_length,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # assume example.text_a is already tokenized by `pre_tokenize.py`
    tokens_a = example.text_a.split(' ')
    tokens_b = example.text_b.split(' ')
    kw_mask_a_raw = copy.deepcopy(example.kw_mask_a_raw)  # `raw` means no padding
    kw_mask_b_raw = copy.deepcopy(example.kw_mask_b_raw)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3,
                           kw_mask_a_raw, kw_mask_b_raw)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # kw_mask_a: 0     0  0    1    1     1       0 0     0  0  0  0   0 0
    # kw_mask_b: 0     0  0    0    0     0       0 0     0  0  0  1   0 0
    # real_mask_a:0     1  1    1    1     1       1 0     0  0  0  0   0 0
    # real_mask_b:0     0  0    0    0     0       0 0     1  1  1  1   1 0
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    kw_mask_a = []
    kw_mask_b = []
    real_mask_a = []
    real_mask_b = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    kw_mask_a.append(0)
    kw_mask_b.append(0)
    real_mask_a.append(0)
    real_mask_b.append(0)

    for idx, token in enumerate(tokens_a):
        tokens.append(token)
        segment_ids.append(0)
        kw_mask_a.append(kw_mask_a_raw[idx])
        real_mask_a.append(1)
        kw_mask_b.append(0)
        real_mask_b.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)
    kw_mask_a.append(0)
    kw_mask_b.append(0)
    real_mask_a.append(0)
    real_mask_b.append(0)

    for idx, token in enumerate(tokens_b):
        tokens.append(token)
        segment_ids.append(1)
        kw_mask_a.append(0)
        real_mask_a.append(0)
        kw_mask_b.append(kw_mask_b_raw[idx])
        real_mask_b.append(1)

    tokens.append("[SEP]")
    segment_ids.append(1)
    kw_mask_a.append(0)
    kw_mask_b.append(0)
    real_mask_a.append(0)
    real_mask_b.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        kw_mask_a.append(0)
        kw_mask_b.append(0)
        real_mask_a.append(0)
        real_mask_b.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(kw_mask_a) == max_seq_length
    assert len(kw_mask_b) == max_seq_length
    assert len(real_mask_a) == max_seq_length
    assert len(real_mask_b) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("real_mask_a: %s" % " ".join([str(x) for x in real_mask_a]))
        tf.logging.info("kw_mask_a: %s" % " ".join([str(x) for x in kw_mask_a]))
        tf.logging.info("real_mask_b: %s" % " ".join([str(x) for x in real_mask_b]))
        tf.logging.info("kw_mask_b: %s" % " ".join([str(x) for x in kw_mask_b]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        kw_mask_a=kw_mask_a,
        kw_mask_b=kw_mask_b,
        real_mask_a=real_mask_a,
        real_mask_b=real_mask_b,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["kw_mask_a"] = create_int_feature(feature.kw_mask_a)
        features["kw_mask_b"] = create_int_feature(feature.kw_mask_b)
        features["real_mask_a"] = create_int_feature(feature.real_mask_a)
        features["real_mask_b"] = create_int_feature(feature.real_mask_b)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "kw_mask_a": tf.FixedLenFeature([seq_length], tf.int64),
        "kw_mask_b": tf.FixedLenFeature([seq_length], tf.int64),
        "real_mask_a": tf.FixedLenFeature([seq_length], tf.int64),
        "real_mask_b": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length,
                       kw_mask_a=None, kw_mask_b=None):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            if kw_mask_a:
                kw_mask_a.pop()
        else:
            tokens_b.pop()
            if kw_mask_b:
                kw_mask_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings,
                 kw_mask_a, kw_mask_b, real_mask_a, real_mask_b, experiment_name=""):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        kw_mask_a=kw_mask_a,
        kw_mask_b=kw_mask_b,
        real_mask_a=real_mask_a,
        real_mask_b=real_mask_b,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    cls_output_layer = model.get_pooled_output()
    # h(cls) + h(A,B)
    if experiment_name == "addhAhB":
        ab_output_layer = model.get_ab_pooled_output()
        output_layer = tf.concat([cls_output_layer, ab_output_layer], -1)
    # h(cls)
    elif experiment_name == "no_kw":
        output_layer = cls_output_layer
    # h(cls) + h_keyword(A,B)
    else:
        kw_output_layer = model.get_kw_pooled_output()
        output_layer = tf.concat([cls_output_layer, kw_output_layer], -1)

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        kw_mask_a = features["kw_mask_a"]
        kw_mask_b = features["kw_mask_b"]
        real_mask_a = features["real_mask_a"]
        real_mask_b = features["real_mask_b"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings,
            kw_mask_a, kw_mask_b, real_mask_a, real_mask_b)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            # to print training loss, code from
            #     https://github.com/google-research/bert/issues/70#issuecomment-436533131
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=200)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                auc = tf.metrics.auc(labels=label_ids, predictions=predictions,
                                     weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


vocab_file = "./keyword_bert/pre_trained/vocab.txt"
output_dir = "./outputs/"
bert_config_file = "./keyword_bert/pre_trained/bert_config_6_layer.json"
init_checkpoint = "./keyword_bert/outputs_lcqmc/model.ckpt-70000"
predict_batch_size = 1
save_checkpoints_steps = 2000
do_lower_case = True
max_seq_length = 128
gpu_device_id = 'cpu'
use_tpu = False
tpu_name = None
iterations_per_loop = 1000

endpoints = "http://localhost:8501/v1/models/keybert:predict"
headers = {"content-type": "application-json"}

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device_id
# tf.logging.set_verbosity(tf.logging.INFO)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

processors = {
    "baike": BaikeProcessor
}

# tokenization.validate_case_matches_checkpoint(do_lower_case, init_checkpoint)

task_name = 'baike'
processor = processors[task_name]()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def get_estimator():
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tpu_cluster_resolver = None
    if use_tpu and tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_name, zone=None, project=None)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=3,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=8,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=2e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=32,
        eval_batch_size=8,
        predict_batch_size=predict_batch_size)


def match_ch(s, kw):
    kw_index = []
    cur_rs = []
    p1 = 0
    p2 = 0
    while (p1 + p2 <= len(s)):
        if p2 == len(kw):  # match succed
            kw_index += cur_rs
            p2 = 0
            p1 += 1
            cur_rs = []
        if (p1 + p2) < len(s) and s[p1 + p2] == kw[p2]:  # matching
            cur_rs.append(p1 + p2)
            p2 += 1
        else:  # match failed
            p2 = 0
            p1 += 1
            cur_rs = []
    return kw_index


def match(s, kws):
    kw_index = set()
    for kw in kws:
        if re.match(r'^[\u4e00-\u9fff]+$', kw):
            kw_index |= set(match_ch(s, kw))
        elif re.match(r'^[a-zA-Z]+$', kw):
            kw_index |= set(match_ch(s, kw))
        else:
            continue
    return kw_index


start_time = time.time()
print("Loading Models...")
index2id = load_js(index2id_path)
# id2titles_df = pd.read_pickle(id2title_df_path)
id2titles_df = pd.read_csv(id2title_path)
print(id2titles_df.shape)
fs = FaissSearch(ann_path=ann_path)
print(f"Loading Models Finished. {round(time.time() - start_time, 2)}")


def jieba_key(x, weight1=2, weight2=1):
    list_keyphrase_tuples = jieba.analyse.extract_tags(x, topK=6, withWeight=True)
    keyphrase_tuples = []
    for tuple in list_keyphrase_tuples:
        if tuple[1] >= weight1:
            keyphrase_tuples.append(tuple)
    if keyphrase_tuples == [] and list_keyphrase_tuples != []:
        if list_keyphrase_tuples[0][1] >= weight2:
            keyphrase_tuples.append(list_keyphrase_tuples[0])
    return keyphrase_tuples


def re_rank(candidate_indexes, candidate_distances, query_text, query_id):
    try:
        id2answer = pd.read_csv(id2answer_path, sep=',')
    except:
        id2answer = pd.read_csv(open(id2answer_path, 'rU'), encoding='utf-8', engine='c')
    id2titles_df = pd.read_csv(id2title_path, sep=',')
    qids = id2titles_df['qid'].tolist()
    index2id = load_js(index2id_path)
    id2category = pd.read_csv(id2category_path, sep=',')

    if verbose:
        print("re_rank")
    candidate_indexes = candidate_indexes[0]
    candidate_distances = candidate_distances[0]
    candidate_items = {}
    if len(query_id.strip()) > 0:
        query_category = id2category[id2category['qid'] == query_id]['category'].values[0]
        for i, candidate_index in enumerate(candidate_indexes[:10]):
            candidate_qid = index2id[str(candidate_index)]
            candidate_category = id2category[id2category['qid'] == candidate_qid]['category'].values[0]
            candidate_answer = id2answer[id2answer['qid'] == candidate_qid]['answer'].values[0]
            # 这里可以设置阈值，但是这里是过滤了跟问题不是同一个category的问题，所以如果还设置阈值的话可能导致返回的数据太少
            if candidate_category == query_category:
                if Set_threshold:
                    if candidate_distances[i] >= threshold:
                        candidate_items.update(
                            {candidate_qid: {'category': candidate_category, 'answer': candidate_answer,
                                             'sim': candidate_distances[i]}})
                else:
                    candidate_items.update({candidate_qid: {'category': candidate_category, 'answer': candidate_answer,
                                                            'sim': candidate_distances[i]}})
    return candidate_items


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def get_token_mask(text):
    jieba_cws = " ".join(jieba.cut(text))
    keywords = [tuple[0] for tuple in jieba_key(jieba_cws)]
    token_cws = " ".join(tokenizer.tokenize(text))
    text_a = token_cws.split()
    kw_a_index = match(text_a, keywords)
    kw_a_mask = [1 if idx in kw_a_index else 0 for idx in range(len(text_a))]
    return token_cws, kw_a_mask


def keybert(query_text, topK=3):
    print("Start Keybert...")
    start_time = time.time()

    bar = ProgressBar('jieba kw')
    token_cws, kw_a_mask = get_token_mask(query_text)
    bar.complete()

    I, D = fs.ann_search_faiss(query=query_text, topn=10)

    bar = ProgressBar('get feature')
    label = '0'
    instances = []
    for i, idx in enumerate(I[0]):
        title = id2titles_df['title'][idx]
        text_b_token_cws, kw_b_mask = get_token_mask(title)
        predict_example = InputExample(i, token_cws, text_b_token_cws, label, kw_a_mask, kw_b_mask)
        feature = convert_single_example(i, predict_example, label_list, max_seq_length, tokenizer)
        instance = {
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids": [feature.segment_ids],
            "label_ids": [feature.label_id],
            "kw_mask_a": [feature.kw_mask_a],
            "kw_mask_b": [feature.kw_mask_b],
            "real_mask_a": [feature.real_mask_a],
            "real_mask_b": [feature.real_mask_b],
            "is_real_example": [feature.is_real_example]
        }
        instances.append(instance)

    # id = 0
    # for row in id2titles_pre_df.itertuples():
    # text_b_token_cws = getattr(row, 'title_token')
    # text_b, kw_b = text_b_token_cws.split(), getattr(row, 'title_keys').split()

    # kw_b_index = match(text_b, kw_b)
    # kw_b_mask = [1 if idx in kw_b_index else 0 for idx in range(len(text_b))]

    # predict_example = InputExample(id, token_cws, text_b_token_cws, label, kw_a_mask, kw_b_mask)
    # feature = convert_single_example(id, predict_example, label_list, max_seq_length, tokenizer)

    # print("feature input Takes ", time.time() - start_time, " s")
    # instance = {
    #     "input_ids": [feature.input_ids],
    #     "input_mask": [feature.input_mask],
    #     "segment_ids": [feature.segment_ids],
    #     "label_ids": [feature.label_id],
    #     "kw_mask_a": [feature.kw_mask_a],
    #     "kw_mask_b": [feature.kw_mask_b],
    #     "real_mask_a": [feature.real_mask_a],
    #     "real_mask_b": [feature.real_mask_b],
    #     "is_real_example": [feature.is_real_example]
    # }
    # instances.append(instance)
    # id += 1
    # print("single prediction input Takes ", time.time() - start_time, " s")
    bar.complete()
    df = pd.DataFrame(columns=['qid', 'probability'])

    bar = ProgressBar('keybert:predict')
    data = json.dumps({"signature_name": "serving_default", "instances": instances})
    response = requests.post(endpoints, data=data, headers=headers)
    prediction = json.loads(response.text)['predictions']
    bar.complete()

    qid_probs = id2titles_df.loc[I[0], ['qid', 'title']]
    qid_probs['probability'] = [x[1] for x in prediction]

    result = qid_probs.nlargest(topK, 'probability')
    end_time = time.time()
    print("Keyword-bert find candidate questions Takes ", end_time - start_time, " s")
    # df = pd.merge(df, id2titles_df, left_index=True, right_index=True).drop(columns=['p_0']).sort_values(by='p_1', ascending=False)
    return result


def get_candidates(query_text):
    # candidate_qids = [index2id[str(item)] for item in I[0]]
    print('d')


if __name__ == '__main__':
    while True:
        # question = '人站在地球上没有头朝下的感觉是因为什么'
        question = input("请输入您的问题：")
        start_t = time.time()
        print("您的问题是：{}".format(question))
        print('正在思考中……')
        results = keybert(question)
        print(results)
        print("Single input (including query search) takes:", time.time() - start_t)
