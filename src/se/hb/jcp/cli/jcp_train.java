// JCP - Java Conformal Prediction framework
// Copyright (C) 2014 - 2016, 2018 - 2019  Anders Gidenstam
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
package se.hb.jcp.cli;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.SortedSet;

import org.json.JSONObject;
import org.json.JSONTokener;

import se.hb.jcp.bindings.smile.GaussianProcessRegressor;
import cern.colt.matrix.DoubleMatrix1D;

import se.hb.jcp.cp.*;
import se.hb.jcp.nc.*;

import se.hb.jcp.ml.ClassifierFactory;
import se.hb.jcp.ml.IClassifier;

import se.hb.jcp.ml.RegressorFactory;
import se.hb.jcp.ml.IRegressor; 
/**
 * Command line training tool for JCP.
 *
 * @author anders.gidenstam(at)hb.se
 */

public class jcp_train
{   
    private boolean _isClassification = false;
    private int     _ncFunctionType = 0;
    private IClassifier _classifier;
    private String  _dataSetFileName;
    private String  _calibrationSetFileName;
    private String  _modelFileName;
    private DataSet _full;
    private DataSet _training;
    private DataSet _calibration;
    private DataSet _test;
    private boolean _useLCCC = false;
    private boolean _useTCC = false;
    private boolean _useCP = true;
    private boolean _useMPC = false;
    private boolean _validate = false;
    private double  _significanceLevel = 0.10;
    private double  _validationFraction = 0.5;
    private double  _calibrationFraction = 0.2;
    
    
    private IRegressor _regressor;
    private boolean _isRegression = false;
    //for transductive conformal regressor : 
    private boolean _useGPP = false; 
    private String _testFileName;

    public jcp_train()
    {
        _training    = new DataSet();
        _calibration = new DataSet();
        _test        = new DataSet();
        _classifier  = new se.hb.jcp.bindings.jlibsvm.SVMClassifier();
    }

    public void run(String[] args) throws IOException {
        processArguments(args);
        if (_isRegression) {
            if(_useGPP) {
                trainGPP(_dataSetFileName);
            }
            else {
                trainRegressor(_dataSetFileName);

            }
        } else {
            if (_useCP && _useTCC) {
                trainTCC(_dataSetFileName);
            } else if (_useCP) {
                if (_calibrationSetFileName != null) {
                    trainICC(_dataSetFileName, _calibrationSetFileName);
                } else {
                    trainICC(_dataSetFileName);
                }
            } else {
                trainPlainClassifier(_dataSetFileName);
            }
        }
    }

    private void processArguments(String[] args) {
        int classifierType = 0;
        int regressorType = 0;
        JSONObject classifierConfig = new JSONObject();
        JSONObject regressorConfig = new JSONObject();

        if (args.length < 1) {
            printUsage();
            System.exit(-1);
        } else {
            for (int i = 0; i < args.length; i++) {
                if (args[i].equals("-h")) {
                    printUsage();
                    System.exit(-1);
                } else if (args[i].equals("-nc")) {
                    if (++i < args.length) {
                        boolean ok = false;
                        try {
                            int c = Integer.parseInt(args[i]);
                            String[] NCFs = ClassificationNonconformityFunctionFactory.getInstance().getNonconformityFunctions();
                            if (0 <= c && c < NCFs.length) {
                                _ncFunctionType = c;
                                ok = true;
                            }
                        } catch (Exception e) {
                        }
                        if (!ok) {
                            System.err.println("Error: Illegal nonconformity function number '" + args[i] + "' given to -nc.");
                            System.err.println();
                            printUsage();
                            System.exit(-1);
                        }
                    }
                } else if (args[i].equals("-c")) {
                    _isClassification = true;
                    if (++i < args.length) {
                        boolean ok = false;
                        try {
                            int c = Integer.parseInt(args[i]);
                            if (0 <= c && c < ClassifierFactory.getInstance().getClassifierTypes().length) {
                                classifierType = c;
                                ok = true;
                            }
                        } catch (Exception e) {
                        }
                        if (!ok) {
                            System.err.println("Error: Illegal classifier number '" + args[i] + "' given to -c.");
                            System.err.println();
                            printUsage();
                            System.exit(-1);
                        }
                    }
                } else if (args[i].equals("-r")) {
                    _isRegression = true;
                    if (++i < args.length) {
                        boolean ok = false;
                        try {
                            int r = Integer.parseInt(args[i]);
                            if (0 <= r && r < RegressorFactory.getInstance().getRegressorTypes().length) {
                                regressorType = r;
                                ok = true;
                            }
                        } catch (Exception e) {
                        }
                        if (!ok) {
                            System.err.println("Error: Illegal regressor number '" + args[i] + "' given to -r.");
                            System.err.println();
                            printUsage();
                            System.exit(-1);
                        }
                    }
                } else if (args[i].equals("-p")) {
                    boolean ok = false;
                    if (++i < args.length) {
                        try {
                            if (_isRegression) {
                                regressorConfig = loadConfig(args[i]);
                            } else {
                                classifierConfig = loadConfig(args[i]);
                            }
                            ok = true;
                        } catch (Exception e) {
                        }
                    }
                    if (!ok) {
                        System.err.println("Error: No or bad configuration file given to -p.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-m")) {
                    if (++i < args.length) {
                        _modelFileName = args[i];
                    } else {
                        System.err.println("Error: No model file name given to -m.");
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
                        }
                        if (!ok) {
                            System.err.println("Error: Illegal significance level '" + args[i] + "' given to -s.");
                            System.err.println();
                            printUsage();
                            System.exit(-1);
                        }
                    } else {
                        System.err.println("Error: No significance level given to -s.");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-v")) {
                    _validate = true;
                } else if (args[i].equals("-tcc")) {
                    _useTCC = true;
                } else if (args[i].equals("-lccc")) {
                    _useLCCC = true;
                } else if (args[i].equals("-mpc")) {
                    _useMPC = true;
                } else if (args[i].equals("-gpp")) {
                    _useGPP = true;
                    if (++i < args.length) {
                        _testFileName = args[i];
                        
                    } else {
                        System.err.println("Error: No model file name given to -gpp (need a test file).");
                        System.err.println();
                        printUsage();
                        System.exit(-1);
                    }
                } else if (args[i].equals("-nc-val")) {
                    if (++i < args.length) {
                        boolean ok = false;
                        try {
                            int c = Integer.parseInt(args[i]);
                            if (0 <= c && c < ClassificationNonconformityFunctionFactory.getInstance().getNonconformityFunctions().length) {
                                _ncFunctionType = c;
                                ok = true;
                            }
                        } catch (Exception e) {
                        }
                        if (!ok) {
                            System.err.println("Error: Illegal nonconformity function number '" + args[i] + "' given to -nc-val.");
                            System.err.println();
                            printUsage();
                            System.exit(-1);
                        }
                    }
                } else if (_dataSetFileName == null) {
                    _dataSetFileName = args[i];
                } else if (_calibrationSetFileName == null) {
                    _calibrationSetFileName = args[i];
                } else {
                    System.err.println("Error: Too many arguments given.");
                    System.err.println();
                    printUsage();
                    System.exit(-1);
                }
            }
        }

        if (_isRegression) {
            _regressor = RegressorFactory.getInstance().createRegressor(regressorType, regressorConfig);
        }
        if (_isClassification) {
            _classifier = ClassifierFactory.getInstance().createClassifier(classifierType, classifierConfig);
        }
    }


    private void printUsage()
    {
        System.out.println
            ("Usage: jcp_train [options] {<libsvm formatted data set>|" +
             "<libsvm formatted training set> " +
             "<libsvm formatted calibration set>}");
        System.out.println("  The supplied data set will be partitioned" +
                           " randomly as needed for the selected");
        System.out.println("  configuration.");
        System.out.println("  Alternatively, for all ICC variants, separate" +
                           " training and calibration data");
        System.out.println("  sets can be supplied and will then be used" +
                           " as is.");
        System.out.println("Options:");
        System.out.println
            ("  -h                Print this message and exit.");
        System.out.println
            ("  -nc <ncfunc #>    Select the nonconformity function to use.");
        System.out.println
            ("                    The following nonconformity functions are" +
             " supported:");
        {
            String[] NCFs =
                ClassificationNonconformityFunctionFactory.getInstance().
                    getNonconformityFunctions();
            for (int i = 0; i < NCFs.length; i++) {
                System.out.print("                      " + i + ". " + NCFs[i]);
                if (i == _ncFunctionType) {
                    System.out.println(" (selected)");
                } else {
                    System.out.println();
                }
            }
        }
        System.out.println
            ("  -c <classifier #> Select the classifier to use.");
        System.out.println
            ("                    The following classifiers are supported:");
        {
            String[] classifiers =
                ClassifierFactory.getInstance().getClassifierTypes();
            for (int i = 0; i < classifiers.length; i++) {
                System.out.println("                      " + i + ". " +
                                   classifiers[i]);
            }
        }
        System.out.println
            ("  -p <config file>  File with configuration parameters for the " +
             "classifier in JSON format.");
        System.out.println
            ("  -m <model file>   Save the created model.");
        System.out.println
            ("  -s <significance> Set the conformal prediction " +
             "significance level for the validation phase (0.0-1.0).");
        System.out.println
            ("  -v                Validate the model after training.");
        System.out.println
            ("                    By default reserves 50% of the data set " +
             "for validation.");
        System.out.println
            ("  -icc              Use inductive conformal classification " +
             "(default).");
        System.out.println
            ("  -tcc              Use transductive conformal classification.");
        System.out.println
            ("  -lccc             Use the label conditional extension to " +
             "conformal classification.");
        System.out.println
            ("  -mpc              Use the multi-probabilistic extension to " +
             "conformal classification. Needs an extra calibration set.");
        System.out.println
            ("  -nocp             Use classification without " +
             "conformal prediction.");
        System.out.println
            ("  -cf <fraction>    Use  <fraction> of the training set for " +
             "calibration (0.0 - 1.0, default 0.2).");
        System.out.println
            ("                    Applies to inductive conformal " +
             "classification.");
        System.out.println
            ("  -vf <fraction>    Use  <fraction> of the data set for " +
             "validation (0.0 - 1.0, default 0.5).");
        System.out.println
            ("                    Applies when doing validation after " +
             "training.");
    }

    private JSONObject loadConfig(String fileName) {
        try (FileInputStream fis = new FileInputStream(fileName)) {
            return new JSONObject(new JSONTokener(fis));
        } catch (Exception e) {
            throw new RuntimeException("Error loading configuration file.", e);
        }
    }
    private void trainRegressor(String dataSetFileName) throws IOException {
        
        long t1 = System.currentTimeMillis();
        _full = DataSetTools.loadDataSet(dataSetFileName);
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");
        if (_validate) {
            splitDataset((1 - _validationFraction)*(1 - _calibrationFraction), (1 - _validationFraction)*_calibrationFraction);
        } else {
            splitDataset((1 - _calibrationFraction), _calibrationFraction);
        }
        DataSet mpcCalibration = null;
        if (_useMPC) {
            DataSet newCalibration = new DataSet();
            DataSet dummy = new DataSet();
            mpcCalibration = new DataSet();
            _calibration.random3Partition(newCalibration, mpcCalibration, dummy, 0.5, 0.5);
            _calibration = newCalibration;
        }
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        System.out.println("Training on " + _training.x.rows() + " instances and calibrating on " + _calibration.x.rows() + " instances.");
        GaussianProcessConformalRegressor icr = new GaussianProcessConformalRegressor(RegressionNonconformityFunctionFactory.getInstance().createNonconformityFunction(_ncFunctionType, _regressor));
        icr.fit(_training.x, _training.y, _calibration.x, _calibration.y);
        if (_validate) {
            RTools.runTest(icr, _test, null, _significanceLevel, false);
            long t4 = System.currentTimeMillis();
            System.out.println("Total Duration " + (double)(t4 - t1)/1000.0 + " sec.");
        }
        
        if (_modelFileName != null) {
            System.out.println("Saving the model to '" + _modelFileName + "'...");
            RTools.saveModel(icr, _modelFileName);
            System.out.println("... Done.");
        }
         
    }
    private void trainGPP(String dataSetFileName) throws IOException {
        long t1 = System.currentTimeMillis();
        _full = DataSetTools.loadDataSet(dataSetFileName);
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");
        long t3 = System.currentTimeMillis();
        _test = DataSetTools.loadDataSet(_testFileName);
        long t4 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");

        //GaussianProcessConformalRegressor gpcr = new GaussianProcessConformalRegressor(RegressionNonconformityFunctionFactory.getInstance().createNonconformityFunction(_ncFunctionType, _regressor));
        /*TestGaussianProcessRegressor gpr = new TestGaussianProcessRegressor(_full.x.toArray(), _full.y, 0.05);
        TestGPRConformalPredictor predictor = new TestGPRConformalPredictor(gpr);
         for (int i = 0; i < _test.x.size(); i++) {
            double[] x_test = _test.x.toArray()[i]; // Point de test
            double y_test = _test.y[i]; // Valeur de test réelle

            // Calcul des intervalles de prédiction conformes
            ArrayList<Double> intervals = predictor.computePredictionIntervals(x_test, 0.05);

            // Affichage des résultats
            System.out.println("Point de test " + (i + 1) + " : x = " + Arrays.toString(x_test) + ", y réel = " + y_test);
            System.out.println("Intervalles de prédiction conformes : " + intervals);
            System.out.println();
        }*/

     
        /* 
        TestGaussianProcessRegressor gpr = new TestGaussianProcessRegressor(new JSONObject().put("sigma", 1.0).put("lambda", 0.1));
        gpr.internalFit(_full.x, _full.y);
        for(int i = 0; i < _test.x.size(); i++) {
            DoubleMatrix1D newPoint = _test.x.viewRow(i);

            // Predict with 95% confidence interval (significance level 0.05)
            double[] interval = gpr.conformalPredict(newPoint, _full.x, _full.y, 0.05);

            System.out.println("Prediction Interval: [" + interval[0] + ", " + interval[1] + "]");
        }*/
        

        
        
    }
    private void trainICC(String dataSetFileName)
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
            splitDataset((1 - _validationFraction)*(1 - _calibrationFraction),
                         (1 - _validationFraction)*_calibrationFraction);
        } else {
            splitDataset((1 - _calibrationFraction), _calibrationFraction);
        }
        DataSet mpcCalibration = null;
        if (_useMPC) {
            DataSet newCalibration = new DataSet();
            DataSet dummy = new DataSet();
            mpcCalibration = new DataSet();
            _calibration.random3Partition(newCalibration, mpcCalibration, dummy, 0.5, 0.5);
            _calibration = newCalibration;
        }
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        trainICC(classes, mpcCalibration, t1, t3);
    }

  private void trainICC(String trainingSetFileName,
                        String calibrationSetFileName)
        throws IOException
    {
        long t1 = System.currentTimeMillis();
        _training = DataSetTools.loadDataSet(trainingSetFileName, _classifier);
        SimpleEntry<double[],SortedSet<Double>> pair =
            DataSetTools.extractClasses(_training);
        double[] classes = pair.getKey();
        SortedSet<Double> classSet = pair.getValue();
        long t2 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t2 - t1)/1000.0 + " sec.");

        _calibration = DataSetTools.loadDataSet(calibrationSetFileName,
                                                _classifier);
        DataSet mpcCalibration = null;
        if (_useMPC) {
            DataSet newCalibration = new DataSet();
            DataSet dummy = new DataSet();
            mpcCalibration = new DataSet();
            _calibration.random3Partition(newCalibration, mpcCalibration, dummy, 0.5, 0.5);
            _calibration = newCalibration;
        }
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        trainICC(classes, mpcCalibration, t1, t3);
    }

  private void trainICC(double[] classes,
                        DataSet  mpcCalibration,
                        long     t1,
                        long     t3)
        throws IOException
    {
        if (!_useMPC) {
            System.out.println("Training on " + _training.x.rows() +
                               " instances and calibrating on " +
                               _calibration.x.rows() +
                               " instances.");
        } else {
            System.out.println("Training on " + _training.x.rows() +
                               " instances and calibrating (underlying + MPC) on " +
                               _calibration.x.rows() + " + " + mpcCalibration.x.rows() +
                               " instances.");
        }

        IConformalClassifier icc =
            new InductiveConformalClassifier
                    (ClassificationNonconformityFunctionFactory.getInstance().
                         createNonconformityFunction(_ncFunctionType,
                                                     classes,
                                                     _classifier),
                     classes, _useLCCC);

        System.out.println("_training.x " + _training.x.getClass().getName() +
                           "(" + _training.x.rows() + "x" +
                           _training.x.columns() + "; " +
                           _training.x.cardinality() + " nonzeros)");
        System.out.println("_calibration.x " +
                           _calibration.x.getClass().getName() +
                           "(" + _calibration.x.rows() + "x" +
                           _calibration.x.columns() + "; " +
                           _calibration.x.cardinality() + " nonzeros)");
        ((InductiveConformalClassifier)icc).fit(_training.x, _training.y,
                                                _calibration.x, _calibration.y);
        if (_useMPC) {
            icc = new se.hb.jcp.cp.ConformalMultiProbabilisticClassifier(icc);
            ((ConformalMultiProbabilisticClassifier)icc)
                .calibrate(mpcCalibration.x, mpcCalibration.y);
        }
        long t4 = System.currentTimeMillis();
        System.out.println("Training complete.");
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");

        if (_validate) {
            CCTools.runTest(icc, _test, null, null, null, _significanceLevel,
                            false);
            long t5 = System.currentTimeMillis();
            System.out.println("Total Duration " + (double)(t5 - t1)/1000.0 +
                               " sec.");
        }

        if (_modelFileName != null) {
            System.out.println("Saving the model to '" +
                               _modelFileName + "'...");
            CCTools.saveModel(icc, _modelFileName);
            System.out.println("... Done.");
        }
    }

  private void trainTCC(String dataSetFileName)
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

        if (!_useMPC) {
            _calibrationFraction = 0.0;
        }
        if (_validate) {
            splitDataset((1 - _validationFraction)*(1 - _calibrationFraction), _calibrationFraction);
        } else {
            splitDataset((1 - _calibrationFraction), _calibrationFraction);
        }
        long t3 = System.currentTimeMillis();
        System.out.println("Duration " + (double)(t3 - t2)/1000.0 + " sec.");

        System.out.println("TCC training set " + _training.x.rows() +
                           " instances.");
        if (_useMPC) {
            System.out.println("MPC calibration set " + _calibration.x.rows() +
                               " instances.");
        }

        IConformalClassifier tcc =
            new TransductiveConformalClassifier
                    (ClassificationNonconformityFunctionFactory.getInstance().
                         createNonconformityFunction(_ncFunctionType,
                                                     classes,
                                                     _classifier),
                     classes, _useLCCC);

        ((TransductiveConformalClassifier)tcc).fit(_training.x, _training.y);
        if (_useMPC) {
            tcc = new se.hb.jcp.cp.ConformalMultiProbabilisticClassifier(tcc);
            ((ConformalMultiProbabilisticClassifier)tcc)
                .calibrate(_calibration.x, _calibration.y);
        }
        long t4 = System.currentTimeMillis();
        System.out.println("Training complete.");
        System.out.println("Duration " + (double)(t4 - t3)/1000.0 + " sec.");

        if (_validate) {
            CCTools.runTest(tcc, _test, null, null, null, _significanceLevel,
                            false);
            long t5 = System.currentTimeMillis();
            System.out.println("Total Duration " + (double)(t5 - t1)/1000.0 +
                               " sec.");
        }

        if (_modelFileName != null) {
            System.out.println("Saving the model to '" +
                               _modelFileName + "'...");
            CCTools.saveModel(tcc, _modelFileName);
            System.out.println("... Done.");
        }
    }

    private void trainPlainClassifier(String dataSetFileName)
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
            splitDataset((1 - _validationFraction), 0.0);
        } else {
            splitDataset(1.0, 0.0);
        }

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

        // Evaluation on the test set.
        if (_validate) {
            System.out.println("Testing accuracy on " + _test.x.rows() +
                               " instances.");

            int correct = 0;
            for (int i = 0; i < _test.x.rows(); i++) {
                double prediction =
                    _classifier.predict(_test.x.viewRow(i));
                if (prediction == _test.y[i]) {
                    correct++;
                }
            }
            long t5 = System.currentTimeMillis();

            System.out.println("Accuracy " +
                               ((double)correct / _test.y.length));
            System.out.println("Duration " +
                               (double)(t5 - t4)/1000.0 + " sec.");
            System.out.println("Total Duration " + (double)(t5 - t1)/1000.0 +
                               " sec.");
        }
        if (_modelFileName != null) {
            System.out.println("Saving the model to '" +
                               _modelFileName + "'...");
            savePlainModel(_classifier, _modelFileName);
            System.out.println("... Done.");
        }
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

    private static void savePlainModel(IClassifier c,
                                       String filename)
        throws IOException
    {
        try (ObjectOutputStream oos =
                 new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(c);
        }
    }

    public static void main(String[] args)
        throws IOException
    {
         new jcp_train().run(args);
    }
}
