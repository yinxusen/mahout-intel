package org.apache.mahout.cf.taste.hadoop.als;

public class PathIndex {
	private int index = -1;

	public int getCurrentIndex() {
		return index;
	}

	public int getNextIndex() {
		return ++index;
	}

	public int getPrevIndex() {
		return index - 1;
	}
}
