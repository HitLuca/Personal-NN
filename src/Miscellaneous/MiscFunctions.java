package Miscellaneous;

import org.jblas.DoubleMatrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static Miscellaneous.NormalizerDenormalizer.denormalize;

/**
 * Created by hitluca on 05/12/15.
 */
public class MiscFunctions {
    public static DoubleMatrix classToMatrix(int c, int n) {
        DoubleMatrix matrix = new DoubleMatrix(n);
        for (int i=0; i<n; i++) {
            matrix.put(i, 0.01);
        }
        matrix.put(c, 0.99);
        return matrix;
    }

    public static void createImage(DoubleMatrix pixels, String filename) {
        BufferedImage bufferedImage = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int denormalized = (int) denormalize(pixels.get(i*28+j), 0, 255);
                int rgb = 255 - denormalized << 16 | 255 - denormalized << 8 | 255 - denormalized;
                bufferedImage.setRGB(i, j, rgb);
            }
        }
        File outputfile = new File(filename);
        try {
            ImageIO.write(bufferedImage, "png", outputfile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
