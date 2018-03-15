#!/usr/bin/env python3

import unittest
from tensorflow import test as tftest
import tensorflow as tf
from tensorflow.python.training.saver import BaseSaverBuilder
import tempfile
import os
import logging
logging.basicConfig(level=logging.INFO)

from optimistic_restore_saver import OptimisticRestoreSaver


class SaveRestoreTestCase(tftest.TestCase):

    def _buildGraph(self, size=1, value=1.):
        for i in range(size):
            tf.Variable(value, name='v_' + str(i))

    def _saveAndRestore(self, save_size=1, restore_size=1, restore_size_2=1, saver_class=tf.train.Saver, builder_class=BaseSaverBuilder):
        with tempfile.TemporaryDirectory() as path:
            fn = 'checkpoint.ckpt'
            afn = os.path.join(path, fn)

            with tf.Graph().as_default():
                graph = tf.get_default_graph()
                self._buildGraph(save_size)
                saver = saver_class(builder=builder_class(), filename=afn, max_to_keep=1)
                self.assertEqual(len(tf.global_variables()), save_size)

                with self.test_session(tf.get_default_graph()) as sess:
                    sess.run(tf.global_variables_initializer())
                    tf.get_default_graph().finalize()
                    saver.save(sess, afn)
                    self.assertIn(fn + '.index', os.listdir(path))

            with tf.Graph().as_default():
                graph = tf.get_default_graph()
                self._buildGraph(restore_size)
                saver = saver_class(builder=builder_class(), filename=afn, max_to_keep=1)
                self.assertEqual(len(tf.global_variables()), restore_size)

                with self.test_session(tf.get_default_graph()) as sess:
                    sess.run(tf.global_variables_initializer())
                    tf.get_default_graph().finalize()
                    saver.restore(sess, afn)
                    saver.save(sess, afn)

            with tf.Graph().as_default():
                graph = tf.get_default_graph()
                self._buildGraph(restore_size_2)
                saver = tf.train.Saver(filename=afn, max_to_keep=1)
                self.assertEqual(len(tf.global_variables()), restore_size_2)

                with self.test_session(tf.get_default_graph()) as sess:
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, afn)


    def testSaveRestore(self):
        self._saveAndRestore(2, 2)

    def testSaveMoreThanRestore(self):
        self._saveAndRestore(3, 2)

    def testSaveLessThanRestore(self):
        try:
            self._saveAndRestore(2, 3)
            self.fail()
        except tf.errors.NotFoundError:
            pass

    def testDynamicSaveRestore(self):
        self._saveAndRestore(2, 2, saver_class=OptimisticRestoreSaver)

    def testDynamicSaveMoreThanRestore(self):
        self._saveAndRestore(3, 2, saver_class=OptimisticRestoreSaver)

    def testDynamicSaveLessThanRestore(self):
        self._saveAndRestore(2, 3, saver_class=OptimisticRestoreSaver)

    def testDynamicSaveLessThanRestoreThenSaveThenLoadNormal(self):
        self._saveAndRestore(2, 3, 3, saver_class=OptimisticRestoreSaver)

    def testDynamicSaveLessThanRestoreThenSaveThenLoadMoreNormal(self):
        try:
            self._saveAndRestore(2, 3, 4, saver_class=OptimisticRestoreSaver)
            self.fail()
        except tf.errors.NotFoundError:
            pass


if __name__ == '__main__':
    # unittest.main()
    tftest.main()
