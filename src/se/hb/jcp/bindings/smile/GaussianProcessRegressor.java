package se.hb.jcp.bindings.smile;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Path;
import java.nio.file.Paths;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import org.json.JSONObject;
import smile.regression.GaussianProcessRegression;
import smile.math.kernel.GaussianKernel;
import smile.math.MathEx;
import se.hb.jcp.ml.RegressorBase;
import se.hb.jcp.ml.IRegressor;

import java.util.Arrays;
import smile.validation.CrossValidation;
import smile.validation.Bag;

public class GaussianProcessRegressor
    extends RegressorBase
    implements Serializable
{
    private static final SparseDoubleMatrix1D _storageTemplate = new SparseDoubleMatrix1D(0);
    protected JSONObject _jsonParameters;
    protected GaussianProcessRegression<double[]> _model;
    protected double _sigma;
    protected double _lambda;

    public GaussianProcessRegressor() {}
    public GaussianProcessRegressor(JSONObject jsonParameters, double sigma, double lambda) {
        this();
        _jsonParameters = jsonParameters;
        _sigma = sigma;
        _lambda = lambda;
    }
    public GaussianProcessRegressor(JSONObject parameters)
    {
        this();
        _jsonParameters = parameters;
        _sigma = parameters.optDouble("sigma", 1.0);
        _lambda = parameters.optDouble("lambda", 0.1);
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) {
        int n = x.rows();
        int d = x.columns();
        double[][] data = new double[n][d];
    
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                data[i][j] = x.get(i, j);
            }
        }
        
        GaussianKernel kernel = new GaussianKernel(_sigma);
        _model = GaussianProcessRegression.fit(data, y, kernel, _lambda);
    }
    
    public IRegressor fitNew(DoubleMatrix2D x, double[] y)
    {
        GaussianProcessRegressor clone = new GaussianProcessRegressor(this.getJsonParameters(), this.getSigma(), this.getLambda());
        clone.fit(x, y);
        return clone;
    }

    @Override
    public double predict(DoubleMatrix1D instance)
    {
        double prediction = _model.predict(instance.toArray());
        System.out.println("Predict : " + prediction);
        return prediction;
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate()
    {
        return _storageTemplate;
    }

    private void writeObject(ObjectOutputStream oos) throws java.io.IOException
    {
        if (_jsonParameters != null) {
            oos.writeObject(_jsonParameters.toString());
        } else {
            oos.writeObject(null);
        }
        oos.writeObject(_model);
        System.out.println("MODEL "  +_model);
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, java.io.IOException
    {
        String jsonText = (String)ois.readObject();
        if (jsonText != null) {
            _jsonParameters = new JSONObject(jsonText);
        }
        _model = (GaussianProcessRegression<double[]>)ois.readObject();
        System.out.println("MODEL "  +_model);
    }
    
    public void gridSearch(DoubleMatrix2D x, double[] y, int folds, double[] sigmaOptions, double[] lambdaOptions) {
        int n = x.rows();
        int d = x.columns();
        double[][] data = new double[n][d];
    
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                data[i][j] = x.get(i, j);
            }
        }

        double bestSigma = 0;
        double bestLambda = 0;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (double sigma : sigmaOptions) {
            for (double lambda : lambdaOptions) {
                GaussianKernel kernel = new GaussianKernel(sigma);
                double score = 0.0;
                
                Bag[] bags = CrossValidation.of(n, folds);

                for (int i = 0; i < folds; i++) {
                    int[] trainIndices = bags[i].samples; 
                    int[] testIndices = bags[i].oob; 

                    double[][] trainData = Arrays.stream(trainIndices)
                                                 .mapToObj(j -> data[j])
                                                 .toArray(double[][]::new);
                    double[] trainLabels = Arrays.stream(trainIndices)
                                                 .mapToDouble(j -> y[j])
                                                 .toArray();
                    double[][] testData = Arrays.stream(testIndices)
                                                .mapToObj(j -> data[j])
                                                .toArray(double[][]::new);
                    double[] testLabels = Arrays.stream(testIndices)
                                                .mapToDouble(j -> y[j])
                                                .toArray();
                    
                    GaussianProcessRegression<double[]> model = GaussianProcessRegression.fit(trainData, trainLabels, kernel, lambda);

                    double foldScore = 0.0;
                    for (int j = 0; j < testData.length; j++) {
                        double prediction = model.predict(testData[j]);
                        foldScore += Math.pow(prediction - testLabels[j], 2);
                    }
                    score += foldScore / testData.length;
                }

                score = -score / folds; 

                if (score > bestScore) {
                    bestScore = score;
                    bestSigma = sigma;
                    bestLambda = lambda;
                }
            }
        }

        System.out.println("Best Sigma: " + bestSigma);
        System.out.println("Best Lambda: " + bestLambda);
        System.out.println("Best Score: " + bestScore);

        _sigma = bestSigma;
        _lambda = bestLambda;
        GaussianKernel bestKernel = new GaussianKernel(_sigma);
        _model = GaussianProcessRegression.fit(data, y, bestKernel, _lambda);
    }

    public void randomSearch(DoubleMatrix2D x, double[] y, int folds, int numIterations, double sigmaMin, double sigmaMax, double lambdaMin, double lambdaMax) {
        int n = x.rows();
        int d = x.columns();
        double[][] data = new double[n][d];
    
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                data[i][j] = x.get(i, j);
            }
        }

        double bestSigma = 0;
        double bestLambda = 0;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < numIterations; i++) {
            double sigma = Math.exp(Math.log(sigmaMin) + Math.random() * (Math.log(sigmaMax) - Math.log(sigmaMin)));
            double lambda = Math.exp(Math.log(lambdaMin) + Math.random() * (Math.log(lambdaMax) - Math.log(lambdaMin)));

            GaussianKernel kernel = new GaussianKernel(sigma);
            double score = 0.0;
            
            Bag[] bags = CrossValidation.of(n, folds);

            for (int j = 0; j < folds; j++) {
                int[] trainIndices = bags[j].samples;
                int[] testIndices = bags[j].oob;

                double[][] trainData = Arrays.stream(trainIndices)
                                             .mapToObj(k -> data[k])
                                             .toArray(double[][]::new);
                double[] trainLabels = Arrays.stream(trainIndices)
                                             .mapToDouble(k -> y[k])
                                             .toArray();
                double[][] testData = Arrays.stream(testIndices)
                                            .mapToObj(k -> data[k])
                                            .toArray(double[][]::new);
                double[] testLabels = Arrays.stream(testIndices)
                                            .mapToDouble(k -> y[k])
                                            .toArray();

                GaussianProcessRegression<double[]> model = GaussianProcessRegression.fit(trainData, trainLabels, kernel, lambda);

                double foldScore = 0.0;
                for (int k = 0; k < testData.length; k++) {
                    double prediction = model.predict(testData[k]);
                    foldScore += Math.pow(prediction - testLabels[k], 2);
                }
                score += foldScore / testData.length;
            }

            score = -score / folds;

            if (score > bestScore) {
                bestScore = score;
                bestSigma = sigma;
                bestLambda = lambda;
            }
        }

        System.out.println("Best Sigma: " + bestSigma);
        System.out.println("Best Lambda: " + bestLambda);
        System.out.println("Best Score: " + bestScore);

        _sigma = bestSigma;
        _lambda = bestLambda;
        GaussianKernel bestKernel = new GaussianKernel(_sigma);
        _model = GaussianProcessRegression.fit(data, y, bestKernel, _lambda);
    }

    public double getLambda() {
        return _lambda;
    }
    public double getSigma() {
        return _sigma;
    }
    public JSONObject getJsonParameters() {
        return _jsonParameters;
    }
}
