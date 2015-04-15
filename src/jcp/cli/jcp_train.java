// Copyright (C) 2014 - 2015  Anders Gidenstam
// License: to be defined.
package jcp.cli;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.AbstractMap.SimpleEntry;
import java.util.SortedSet;
import java.util.Random;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.ObjectMatrix2D;

import org.json.JSONObject;
import org.json.JSONTokener;

import jcp.cp.*;
import jcp.nc.*;
import jcp.io.*;

import jcp.ml.ClassifierFactory;
import jcp.ml.IClassifier;

/**
 * Command line training tool for JCP.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class jcp_train
{
    private IClassifier _classifier;
    private String  _dataSetFileName;
    private String  _modelFileName;
    private DataSet _full;
    private DataSet _training;
    private DataSet _calibration;
    private DataSet _test;
    private boolean _useCP = true;
    private boolean _validate = false;
    private double  _significanceLevel = 0.10;

    public jcp_train()
    {
        _training    = new DataSet();
        _calibration = new DataSet();
        _test        = new DataSet();
        _classifier  = new jcp.bindings.libsvm.SVMClassifier();
    }

    public void run(String[] args)
        throws IOException
    {
        processArguments(args);
        if (_useCP) {
            // Supports train, calibrate and save and/or test.
            runICCTest(_dataSetFileName);
        } else {
            // FIXME: Only initial use case yet. Train & test.
            runPlainSVMTest(_dataSetFileName);
        }
    }

    private void processArguments(String[] args)
    {
        int classifierType = 0;
        JSONObject classifierConfig = new JSONObject();

        // Load and create training and calibration sets.
        if (args.length < 1) {
            printUsage();
            System.exit(-1);
        } else {
            for (int i = 0; i < args.length; i++) {
                if (args[i].equals("-h")) {
                    printUsage();
                    System.exit(-1);
                } else if (args[i].equals("-c")) {
                    if (++i < args.length) {
                        boolean ok = false;
                        try {
                            int c = Integer.parseInt(args[i]);
                            if (0 <= c &&
                                c < ClassifierFactory.getInstance().
                                        getClassifierTypes().length) {
                                classifierType = c;
                                ok = true;
                            }
                        } catch (Exception e) {
                            // Handled below as ok is false.
                        }
                        if (!ok) {
                            System.err.println
                                ("Error: Illegal classifier number '" +
                                 args[i] +
                                 "' given to -c.");
                            System.err.println();
                            printUsage();
                            System.exit(-1);
                        }
                    }
                } else if (args[i].equals("-p")) {
                    boolean ok = false;
                    if (++i < args.length) {
                        try {
                            classifierConfig = loadClassifierConfig(args[i]);
                            ok = true;
                        } catch (Exception e) {
                            // Handled below as ok is false.
                        }
                    }
                    if (!ok) {
                        System.err.println
                            ("Error: No or bad configuration file given " +
                             "to -p.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-m")) {
                    if (++i < args.length) {
                        _modelFileName = args[i];
                    } else {
                        System.err.println
                            ("Error: No model file name given to -m.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-s")) {
                    if (++i < args.length) {
                        boolean ok = false;
                        try {
                            double s = Double.parseDouble(args[i]);
                            if (0.0 <= s && s <= 1.0) {
                                _significanceLevel = s;
                                ok = true;
                            }
                        } catch (Exception e) {
                            // Handled below as ok is false.
                        }
                        if (!ok) {
                            System.err.println
                                ("Error: Illegal significance level '" +
                                 args[i] +
                                 "' given to -s.");
                            System.err.println();
                            printUsage();
                            System.exit(-1);
                        }
                    } else {
                        System.err.println
                            ("Error: No significance level given to -s.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-v")) {
                    _validate = true;
                } else if (args[i].equals("-nocp")) {
                    _useCP = false;
                } else {
                    // The last unknown argument should be the dataset file.
                    _dataSetFileName = args[i];
                }
            }
        }
        if (_dataSetFileName == null) {
            System.err.println
                ("Error: No data set file name given.");
            System.err.println();
            printUsage();
            System.exit(-1);
        }
        _classifier =
            ClassifierFactory.getInstance().createClassifier(classifierType,
                                                             classifierConfig);
    }

    private void printUsage()
    {
        System.out.println
            ("Usage: jcp_train [options] <libsvm formatted data set>");
        System.out.println();
        System.out.println
            ("  -h                Print this message and exit.");
        System.out.println
            ("  -c <classifier #> Select the classifier to use.");
        System.out.println
            ("                    The following classifiers are supported.");
        {
            int i = 0;
            for (String c : ClassifierFactory.getInstance().
                                getClassifierTypes()) {
                System.out.println("                      " + i + ". " + c);
                i++;
            }
        }
        System.out.println
            ("  -p <config file>  File with configuration parameters for the " +
             "classifier in JSON format.");
        System.out.println
            ("  -m <model file>   Save the created model.");
        System.out.println
            ("  -s <significance> Set the conformal prediction " +
             "significance level for the test phase (0.0-1.0).");
        System.out.println
            ("  -v                Validate the model after training.");
        System.out.println
            ("                    Reserves 50% of the data set for " +
             "validation.");
        System.out.println
            ("  -nocp             Test classification without " +
             "conformal prediction.");
    }

    private void runICCTest(String dataSetFileName)
        throws IOException
    {
        long t1 = System.currentTimeMillis();
        _full = DataSetTools.loadDataSet(dataSetFileName);
        SimpleEntry<double[],SortedSet<Double>> pair =
            DataSetTools.extractClasses(_full);
        double[] classes = pair.getKey();
        SortedSet<Double> classSet = pair.getValue();
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");

        if (_validate) {
            splitDataset(0.4, 0.1);
        } else {
            splitDataset(0.8, 0.2);
        }
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        System.out.println("Training on " + _training.x.rows() +
                           " instances and calibrating on " +
                           _calibration.x.rows() +
                           " instances.");

        InductiveConformalClassifier icc =
            new InductiveConformalClassifier(classes);
        icc._nc =
            new ClassProbabilityNonconformityFunction(classes, _classifier);

        icc.fit(_training.x, _training.y, _calibration.x, _calibration.y);
        long t4 = System.currentTimeMillis();
        System.out.println("Training complete.");
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");

        if (_validate) {
            ICCTools.runTest(icc, _test, null, _significanceLevel);
            long t5 = System.currentTimeMillis();
            System.out.println("Total Duration " + (double)(t5 - t1)/1000.0 +
                               " sec.");
        }

        if (_modelFileName != null) {
            System.out.println("Saving the model to '" +
                               _modelFileName + "'...");
            ICCTools.saveModel(icc, _modelFileName);
            System.out.println("... Done.");
        }
    }

    private void runPlainSVMTest(String dataSetFileName)
        throws IOException
    {
        long t1 = System.currentTimeMillis();
        _full = DataSetTools.loadDataSet(dataSetFileName);
        SimpleEntry<double[],SortedSet<Double>> pair =
            DataSetTools.extractClasses(_full);
        double[] classes = pair.getKey();
        SortedSet<Double> classSet = pair.getValue();
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");

        splitDataset(0.4, 0.0);
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        System.out.println("Training on " + _training.x.rows() +
                           " instances and calibrating on " +
                           _calibration.x.rows() +
                           " instances.");

        _classifier.fit(_training.x, _training.y);
        long t4 = System.currentTimeMillis();
        System.out.println("Training complete.");
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");

        System.out.println("Testing accuracy on " + _test.x.rows() +
                           " instances.");

        // Evaluation on the test set.
        int correct = 0;
        for (int i = 0; i < _test.x.rows(); i++) {
            double prediction =
                _classifier.predict(_test.x.viewRow(i));
            if (prediction == _test.y[i]) {
                correct++;
            }
        }
        long t5 = System.currentTimeMillis();

        System.out.println("Accuracy " + ((double)correct / _test.y.length));
        System.out.println("Duration " + (double)(t5 - t4)/1000.0 + " sec.");
        System.out.println("Total Duration " + (double)(t5 - t1)/1000.0 +
                           " sec.");
    }

    private void splitDataset(double trainingFraction,
                              double calibrationFraction)
    {
        // Set template matrix types for the split data set.
        _training.x = _classifier.nativeStorageTemplate().like2D(0,0);
        _calibration.x = _training.x;
        _test.x = _training.x;
        // Partition the full data set into training, calibartion and test sets.
        _full.random3Partition(_training, _calibration, _test,
                               trainingFraction, calibrationFraction);
        System.out.println("Split the data set into training set, " +
                           _training.x.rows() + " instances, " +
                           "calibration set, " +
                           _calibration.x.rows() + " instances, " +
                           "and test set, " + _test.x.rows() + " instances.");
    }

    private JSONObject loadClassifierConfig(String filename)
        throws IOException
    {
        try (FileInputStream fis = new FileInputStream(filename)) {
            return new JSONObject(new JSONTokener(fis));
        } catch (Exception e) {
            System.err.println
                ("Error: Failed to load classifier configuration from '" +
                 filename + "'.");
            throw e;
        }
    }

    public static void main(String[] args)
        throws IOException
    {
         new jcp_train().run(args);
    }
}
