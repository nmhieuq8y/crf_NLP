import java.io.IOException;

public class main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			CRF crf = new CRF();
			/* crf.learn(); */

			crf.test();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
