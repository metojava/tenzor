package com.tenzor;

import com.tenzor.core.Tensor;
import com.tenzor.core.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TensorTest {

    @Test
    public void testScalar() {
        Tensor t = new Tensor(5.0);
        assertEquals(5.0, t.scalar(), 1e-10);
        assertEquals(1, t.size());
    }

    @Test
    public void test1D() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5});
        assertEquals(5, t.size());
        assertEquals(3.0, t.get(2), 1e-10);
    }

    @Test
    public void test2D() {
        Tensor t = new Tensor(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        assertEquals(6, t.size());
        assertEquals(4, t.get(1, 1), 1e-10);
    }

    @Test
    public void testZeros() {
        Tensor z = Tensor.zeros(3, 4);
        assertEquals(12, z.size());
        for (double d : z.getData()) assertEquals(0.0, d, 1e-10);
    }

    @Test
    public void testOnes() {
        Tensor o = Tensor.ones(5);
        assertEquals(5, o.size());
        for (double d : o.getData()) assertEquals(1.0, d, 1e-10);
    }

    @Test
    public void testEye() {
        Tensor eye = Tensor.eye(3);
        assertEquals(1.0, eye.get(0, 0), 1e-10);
        assertEquals(0.0, eye.get(0, 1), 1e-10);
        assertEquals(1.0, eye.get(1, 1), 1e-10);
        assertEquals(1.0, eye.get(2, 2), 1e-10);
    }

    @Test
    public void testLinspace() {
        Tensor l = Tensor.linspace(0, 1, 11);
        assertEquals(11, l.size());
        assertEquals(0.0, l.get(0), 1e-10);
        assertEquals(0.5, l.get(5), 1e-10);
        assertEquals(1.0, l.get(10), 1e-10);
    }

    @Test
    public void testArange() {
        Tensor a = Tensor.arange(5);
        assertEquals(5, a.size());
        assertEquals(0, a.get(0), 1e-10);
        assertEquals(4, a.get(4), 1e-10);
    }

    @Test
    public void testAdd() {
        Tensor a = new Tensor(new double[]{1, 2, 3});
        Tensor b = new Tensor(new double[]{4, 5, 6});
        Tensor c = a.add(b);
        assertEquals(5.0, c.get(0), 1e-10);
        assertEquals(7.0, c.get(1), 1e-10);
        assertEquals(9.0, c.get(2), 1e-10);
    }

    @Test
    public void testSub() {
        Tensor a = new Tensor(new double[]{10, 20, 30});
        Tensor b = new Tensor(new double[]{1, 2, 3});
        Tensor c = a.sub(b);
        assertEquals(9.0, c.get(0), 1e-10);
        assertEquals(18.0, c.get(1), 1e-10);
        assertEquals(27.0, c.get(2), 1e-10);
    }

    @Test
    public void testMul() {
        Tensor a = new Tensor(new double[]{2, 3, 4});
        Tensor b = new Tensor(new double[]{5, 6, 7});
        Tensor c = a.mul(b);
        assertEquals(10.0, c.get(0), 1e-10);
        assertEquals(18.0, c.get(1), 1e-10);
        assertEquals(28.0, c.get(2), 1e-10);
    }

    @Test
    public void testMatmul() {
        Tensor a = new Tensor(new double[][]{{1, 2}, {3, 4}});
        Tensor b = new Tensor(new double[][]{{5, 6}, {7, 8}});
        Tensor c = a.matmul(b);
        assertEquals(19.0, c.get(0, 0), 1e-10);
        assertEquals(22.0, c.get(0, 1), 1e-10);
        assertEquals(43.0, c.get(1, 0), 1e-10);
        assertEquals(50.0, c.get(1, 1), 1e-10);
    }

    @Test
    public void testTranspose() {
        Tensor a = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor b = a.transpose();
        assertEquals(1.0, b.get(0, 0), 1e-10);
        assertEquals(4.0, b.get(0, 1), 1e-10);
        assertEquals(2.0, b.get(1, 0), 1e-10);
        assertEquals(3.0, b.get(2, 0), 1e-10);
    }

    @Test
    public void testSigmoid() {
        Tensor t = new Tensor(new double[]{0, 1, -1});
        Tensor s = t.sigmoid();
        assertEquals(0.5, s.get(0), 1e-6);
        assertTrue(s.get(1) > 0.5);
        assertTrue(s.get(2) < 0.5);
    }

    @Test
    public void testRelu() {
        Tensor t = new Tensor(new double[]{-2, -1, 0, 1, 2});
        Tensor r = t.relu();
        assertEquals(0.0, r.get(0), 1e-10);
        assertEquals(0.0, r.get(1), 1e-10);
        assertEquals(0.0, r.get(2), 1e-10);
        assertEquals(1.0, r.get(3), 1e-10);
        assertEquals(2.0, r.get(4), 1e-10);
    }

    @Test
    public void testSoftmax() {
        Tensor t = new Tensor(new double[]{1, 2, 3});
        Tensor s = t.softmax();
        double sum = 0;
        for (double d : s.getData()) sum += d;
        assertEquals(1.0, sum, 1e-6);
    }

    @Test
    public void testReshape() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5, 6});
        Tensor r = t.reshape(2, 3);
        assertEquals(2, r.shape().dim(0));
        assertEquals(3, r.shape().dim(1));
        assertEquals(1.0, r.get(0, 0), 1e-10);
        assertEquals(6.0, r.get(1, 2), 1e-10);
    }

    @Test
    public void testSum() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5});
        assertEquals(15.0, t.sum().scalar(), 1e-10);
    }

    @Test
    public void testMean() {
        Tensor t = new Tensor(new double[]{2, 4, 6, 8});
        assertEquals(5.0, t.mean().scalar(), 1e-10);
    }

    @Test
    public void testArgmax() {
        Tensor t = new Tensor(new double[]{1, 5, 3, 2, 4});
        assertEquals(1.0, t.argmax().scalar(), 1e-10);
    }

    @Test
    public void testBroadcasting() {
        Tensor a = new Tensor(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor b = new Tensor(new double[]{10, 20, 30});
        Tensor c = a.add(b);
        assertEquals(11.0, c.get(0, 0), 1e-10);
        assertEquals(22.0, c.get(0, 1), 1e-10);
        assertEquals(33.0, c.get(0, 2), 1e-10);
        assertEquals(14.0, c.get(1, 0), 1e-10);
    }
}
