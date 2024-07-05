package se.hb.jcp.bindings.deeplearning4j;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.nd4j.linalg.learning.config.Nesterovs;

import smile.validation.CrossValidation;
import smile.validation.Bag;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;

import org.json.JSONObject;

import se.hb.jcp.ml.RegressorBase;
import se.hb.jcp.ml.IRegressor;

public class NN4jRegressor extends RegressorBase implements IRegressor, java.io.Serializable {
    private static final SparseDoubleMatrix1D _storageTemplate = new SparseDoubleMatrix1D(0);
    protected MultiLayerNetwork _model;

    public NN4jRegressor() {}

    public NN4jRegressor(JSONObject configuration) {
        this();
        _model = createNetworkFromConfig(configuration);
    }

    public NN4jRegressor(String modelFilePath) {
        this();
        try {
            _model = MultiLayerNetwork.load(new File(modelFilePath), true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public NN4jRegressor(MultiLayerNetwork model) {
        _model = model;
    }

    @Override
    public IRegressor fitNew(DoubleMatrix2D x, double[] y) {
        NN4jRegressor newRegressor = new NN4jRegressor();
        newRegressor.internalFit(x, y);
        return newRegressor;
    }

    @Override
    public double predict(DoubleMatrix1D instance) {
        INDArray input = Nd4j.create(instance.toArray()).reshape(1, instance.size());
        INDArray output = _model.output(input);
        System.out.println("Prediction " + output);
        return output.getDouble(0);
    }

    @Override
    public DoubleMatrix1D nativeStorageTemplate() {
        return _storageTemplate;
    }

    @Override
    protected void internalFit(DoubleMatrix2D x, double[] y) {
        _model = createAndTrainNetwork(x.toArray(), y, 50, 0.1, 50);
    }

    public void gridSearch(DoubleMatrix2D x, double[] y, int folds, double[] learningRateOptions, int[] hiddenLayerSizes, int[] numEpochsOptions) {
        int n = x.rows();
        int d = x.columns();
        double[][] data = new double[n][d];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                data[i][j] = x.get(i, j);
            }
        }

        double bestLearningRate = 0;
        int bestNumEpochs = 0;
        int[] bestHiddenLayerSizes = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (double learningRate : learningRateOptions) {
            for (int numEpochs : numEpochsOptions) {
                for (int hiddenLayerSize : hiddenLayerSizes) {
                    MultiLayerNetwork model = createAndTrainNetwork(data, y, hiddenLayerSize, learningRate, numEpochs);

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

                        DataSet trainSet = createDataSet(trainData, trainLabels);
                        DataSet testSet = createDataSet(testData, testLabels);

                        model.fit(trainSet);

                        double foldScore = 0.0;
                        INDArray testFeatures = Nd4j.create(testData);
                        INDArray testPredictions = model.output(testFeatures);

                        for (int j = 0; j < testData.length; j++) {
                            foldScore += Math.pow(testPredictions.getDouble(j) - testLabels[j], 2);
                        }
                        score += foldScore / testData.length;
                    }

                    score = -score / folds;  

                    if (score > bestScore) {
                        bestScore = score;
                        bestLearningRate = learningRate;
                        bestNumEpochs = numEpochs;
                        bestHiddenLayerSizes = new int[]{hiddenLayerSize};
                    }
                }
            }
        }

        System.out.println("Best Learning Rate: " + bestLearningRate);
        System.out.println("Best Number of Epochs: " + bestNumEpochs);
        System.out.println("Best Hidden Layer Sizes: " + Arrays.toString(bestHiddenLayerSizes));
        System.out.println("Best Score: " + bestScore);

        _model = createAndTrainNetwork(data, y, bestHiddenLayerSizes[0], bestLearningRate, bestNumEpochs);
    }

    private MultiLayerNetwork createAndTrainNetwork(double[][] x, double[] y, int hiddenLayerSize, double learningRate, int nEpochs) {
        int inputFeatures = x[0].length;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(learningRate, 0.9))
            .l2(1e-4)
            .list()
            .layer(new DenseLayer.Builder().nIn(inputFeatures).nOut(hiddenLayerSize).build())
            .layer(new DenseLayer.Builder().nIn(hiddenLayerSize).nOut(hiddenLayerSize).build())
            .layer(new DenseLayer.Builder().nIn(hiddenLayerSize).nOut(hiddenLayerSize).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(hiddenLayerSize).nOut(1).build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        DataSet dataSet = createDataSet(x, y);
        for (int i = 0; i < nEpochs; i++) {
            model.fit(dataSet);
        }
        
        return model;
    }

    private DataSet createDataSet(double[][] x, double[] y) {
        int rows = x.length;
        INDArray features = Nd4j.create(x);
        INDArray labels = Nd4j.create(y, new int[]{rows, 1});
        DataSet dataSet = new DataSet(features, labels);
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);
        return dataSet;
    }
    @Override
    public NN4jRegressor clone() {
        try {
            // Serialize and then deserialize to achieve deep cloning
            ByteArrayOutputStream byteOut = new ByteArrayOutputStream();
            ObjectOutputStream out = new ObjectOutputStream(byteOut);
            out.writeObject(this);
            ByteArrayInputStream byteIn = new ByteArrayInputStream(byteOut.toByteArray());
            ObjectInputStream in = new ObjectInputStream(byteIn);
            return (NN4jRegressor) in.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    private MultiLayerNetwork createNetworkFromConfig(JSONObject config) {
        return null;
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        if (_model != null) {
            String fileName = Long.toHexString(Double.doubleToLongBits(Math.random())) + ".deeplearning4j";
            _model.save(new File(fileName));
            oos.writeObject(fileName);
        } else {
            oos.writeObject(null);
        }
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        String fileName = (String) ois.readObject();
        if (fileName != null) {
            _model = MultiLayerNetwork.load(new File(fileName), true);
        }
    }
}
