import java.io.IOException;

import org.apache.nlp.sequencelearning.crf.CRFDriver;
import org.apache.nlp.sequencelearning.crf.CRFModel;

public class CRF {

	static String templfile = "C:\\Users\\nmhie\\Desktop\\crf_NLP\\src\\test\\files\\template";
	// static String trainfile =
	// "C:\\Users\\nmhie\\Desktop\\crf_NLP\\src\\test\\files\\train.data";
	// static String testfile =
	// "C:\\Users\\nmhie\\Desktop\\crf_NLP\\src\\test\\files\\test.data";
	static String modelfile = "C:\\Users\\nmhie\\Desktop\\crf_NLP\\src\\test\\files\\model.data";
	static String trainfile = "C:\\Users\\nmhie\\Desktop\\crf_NLP\\src\\test\\files\\TrainSet.pos";
	static String testfile = "C:\\Users\\nmhie\\Desktop\\crf_NLP\\src\\test\\files\\TestSet.pos";
	static CRFModel model;
	CRFDriver driver;

	public void learn() throws IOException {
		System.out.println("Running CRF learn...");
		driver = new CRFDriver();

		boolean textmodelfile = false;
		int xsize = 2;
		int maxitr = 10000;
		double freq = 1.0;
		double eta = 0.0001;
		double C = 1.0;
		String algorithm = "L-BFGS";
		model = driver.crf_learn(templfile, trainfile, modelfile, textmodelfile, xsize, maxitr, freq, eta, C,
				algorithm);
		model.writeModel(modelfile);
	}

	public void test() throws IOException {
		System.out.println("\nRunning CRF test...");
		driver = new CRFDriver();
		int xsize = 2;
		model = new CRFModel();
		model.readModel(modelfile);
		driver.crf_test(templfile, testfile, model, xsize);
	}

}
