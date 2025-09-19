package ir.vsr;

import java.util.*;
import ir.utilities.*;

public class FeedbackProx {
  public static double ALPHA = 1;
  public static double BETA = 1;
  public static double GAMMA = 1;

  public HashMapVector queryVector;
  public Retrieval[] retrievals;
  public InvertedPosIndex index;
  public ArrayList<DocumentReference> goodDocRefs = new ArrayList<>();
  public ArrayList<DocumentReference> badDocRefs = new ArrayList<>();

  public FeedbackProx(HashMapVector queryVector, Retrieval[] retrievals, InvertedPosIndex index) {
    this.queryVector = queryVector;
    this.retrievals = retrievals;
    this.index = index;
  }

  public void addGood(DocumentReference docRef) { goodDocRefs.add(docRef); }
  public void addBad(DocumentReference docRef) { badDocRefs.add(docRef); }
  public boolean isEmpty() { return goodDocRefs.isEmpty() && badDocRefs.isEmpty(); }

  public void getFeedback(int showNumber) {
    DocumentReference docRef = retrievals[showNumber - 1].docRef;
    String response = UserInput.prompt("Is document #" + showNumber + ":" + docRef.file.getName() +
        " relevant (y:Yes, n:No, u:Unsure)?: ");
    if (response.equals("y"))
      goodDocRefs.add(docRef);
    else if (response.equals("n"))
      badDocRefs.add(docRef);
    else if (!response.equals("u"))
      getFeedback(showNumber);
  }

  public boolean haveFeedback(int showNumber) {
    DocumentReference docRef = retrievals[showNumber - 1].docRef;
    return goodDocRefs.contains(docRef) || badDocRefs.contains(docRef);
  }

  public HashMapVector newQuery() {
    HashMapVector newQuery = queryVector.copy();
    newQuery.multiply(ALPHA / newQuery.maxWeight());
    for (DocumentReference docRef : goodDocRefs) {
      Document doc = docRef.getDocument(index.docType, index.stem);
      HashMapVector vector = doc.hashMapVector();
      vector.multiply(BETA / vector.maxWeight());
      newQuery.add(vector);
    }
    for (DocumentReference docRef : badDocRefs) {
      Document doc = docRef.getDocument(index.docType, index.stem);
      HashMapVector vector = doc.hashMapVector();
      vector.multiply(GAMMA / vector.maxWeight());
      newQuery.subtract(vector);
    }
    return newQuery;
  }
}