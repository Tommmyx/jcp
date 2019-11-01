// JCP - Java Conformal Prediction framework
// Copyright (C) 2016, 2019  Anders Gidenstam
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
package se.hb.jcp.ml;

import java.util.SortedSet;
import java.util.TreeSet;

import cern.colt.matrix.DoubleMatrix2D;

/**
 * Base class for classifiers that provide implementations of some of the
 * generic IClassifierInformation methods.
 */
public abstract class ClassifierBase
    implements IClassifier,
               java.io.Serializable
{
    private int _attributeCount = -1;
    private Double[] _labels = null;
    private boolean _trained = false;

    @Override
    public final void fit(DoubleMatrix2D x, double[] y)
    {
        // Setup remaining classifier information.
        _attributeCount = x.columns();
        // FIXME: Extracting the set of labels from y is rather expensive
        //        so it might be good to do that once only for a data set
        //        and then pass it around to where it is needed.
        SortedSet<Double> uniqueLabels = new TreeSet<Double>();
        for (int i = 0; i < y.length; i++) {
            uniqueLabels.add(y[i]);
        }
        _labels = uniqueLabels.toArray(new Double[0]);

        // Train the model.
        internalFit(x, y);
        _trained = true;
    }

    @Override
    public final boolean isTrained()
    {
        return _trained;
    }

    @Override
    public final int getAttributeCount()
    {
        return _attributeCount;
    }

    @Override
    public final Double[] getLabels()
    {
        return _labels;
    }

    protected abstract void internalFit(DoubleMatrix2D x, double[] y);
}
