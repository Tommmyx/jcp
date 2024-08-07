// JCP - Java Conformal Prediction framework
// Copyright (C) 2014  Henrik Linusson
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
package se.hb.jcp.cp;

import se.hb.jcp.util.ParallelizedAction;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.SortedMap;

import se.hb.jcp.nc.IClassificationNonconformityFunction;
import se.hb.jcp.nc.IRegressionNonconformityFunction;

public class QuantileConformalRegressor implements IConformalRegressor, java.io.Serializable {
    
    private static final boolean PARALLEL = true;

    private IRegressionNonconformityFunction _nc;
    private double[] _calibrationScores;

    public QuantileConformalRegressor(IRegressionNonconformityFunction nc) {
        _nc = nc;
    }

    public void fit(DoubleMatrix2D xtr, double[] ytr, DoubleMatrix2D xcal, double[] ycal) {
        _nc.fit(xtr, ytr);
        calibrate(xcal, ycal);
    }

    public void calibrate(DoubleMatrix2D xcal, double[] ycal) {
        int n = xcal.rows();
        _calibrationScores = new double[n];
    
        for (int i = 0; i < n; i++) {
            DoubleMatrix1D instance = xcal.viewRow(i);
            double label = ycal[i];
            _calibrationScores[i] = _nc.calculateNonConformityScore(instance, label);
        }
    
        Arrays.sort(_calibrationScores);
    
        // Print calibration scores for debugging
        System.out.println("Calibration Scores: " + Arrays.toString(_calibrationScores));
    }
       

    public double calculateQuantile(double alpha) {
        int n = _calibrationScores.length;
        int index = (int) Math.ceil((1 - alpha) * (n + 1)) - 1;
        System.out.println("qhat" + _calibrationScores[Math.max(0, Math.min(index, n - 1))]);
        return _calibrationScores[Math.max(0, Math.min(index, n - 1))];
    }


    public double[] predictIntervals(DoubleMatrix1D x, double alpha) {
        double [] prediction = _nc.predictWithUncertainty(x);
        double predictedValue = prediction[0];
        double uncertainty = prediction[1];
        double qhat = calculateQuantile(alpha);
        System.out.println("predicted : " + predictedValue + " uncert " + uncertainty);
        return new double[]{predictedValue - qhat * uncertainty, predictedValue + qhat * uncertainty};
    }

    public double[][] predictIntervals(DoubleMatrix2D x, double confidence) {
        int n = x.rows();
        double[][] intervals = new double[n][2];
        if (!PARALLEL) {
            for (int i = 0; i < n; i++) {
                intervals[i] = predictIntervals(x.viewRow(i), confidence);
            }
        } else {
            PredictIntervalsAction all = new PredictIntervalsAction(x, intervals, confidence, 0, n);
            all.start();
        }
        return intervals;
    }

    @Override
    public boolean isTrained() {
        return _calibrationScores != null;
    }

    @Override
    public int getAttributeCount() {
        if (getNonconformityFunction() != null) {
            return getNonconformityFunction().getAttributeCount();
        } else {
            return -1;
        }
    }

     @Override
    public DoubleMatrix1D nativeStorageTemplate()
    {
        if (getNonconformityFunction() != null) {
            return getNonconformityFunction().nativeStorageTemplate();
        } else {
            return new cern.colt.matrix.impl.SparseDoubleMatrix1D(0);
        }
    }

    private void writeObject(ObjectOutputStream oos)
        throws java.io.IOException
    {
        oos.writeObject(_nc);
        oos.writeObject(_calibrationScores);
    }

    @SuppressWarnings("unchecked") // There is not much to do if the saved
                                   // value doesn't match the expected type.
    private void readObject(ObjectInputStream ois)
        throws ClassNotFoundException, java.io.IOException
    {
        _nc = (IRegressionNonconformityFunction)ois.readObject();
        _calibrationScores = (double[])ois.readObject();
    }


    public IRegressionNonconformityFunction getNonconformityFunction() {
        return _nc;
    }
    class PredictIntervalsAction extends ParallelizedAction {
        DoubleMatrix2D _x;
        double[][] _intervals;
        double _confidence;

        public PredictIntervalsAction(DoubleMatrix2D x, double[][] intervals, double confidence, int first, int last) {
            super(first, last);
            _x = x;
            _intervals = intervals;
            _confidence = confidence;
        }

        @Override
        protected void compute(int i) {
            _intervals[i] = predictIntervals(_x.viewRow(i), _confidence);
        }

        @Override
        protected ParallelizedAction createSubtask(int first, int last) {
            return new PredictIntervalsAction(_x, _intervals, _confidence, first, last);
        }
    }

    class CalculateNCScoresAction extends ParallelizedAction {
        DoubleMatrix2D _x;
        double[] _y;
        double[] _nonConformityScores;

        public CalculateNCScoresAction(DoubleMatrix2D x, double[] y, double[] nonConformityScores, int first, int last) {
            super(first, last);
            _x = x;
            _y = y;
            _nonConformityScores = nonConformityScores;
        }

        @Override
        protected void compute(int i) {
            DoubleMatrix1D instance = _x.viewRow(i);
            _nonConformityScores[i] = _nc.calculateNonConformityScore(instance, _y[i]);
        }

        @Override
        protected ParallelizedAction createSubtask(int first, int last) {
            return new CalculateNCScoresAction(_x, _y, _nonConformityScores, first, last);
        }
    }

}
