package ir.vsr;

import java.io.*;

/**
 * Text document reader for positional indexing.
 *
 * This class behaves like {@link TextFileDocument} with tokenization, optional stemming,
 * and filtering of non-letter tokens, but intentionally does not remove stopwords.
 * Retaining stopwords ensures token distances remain accurate for proximity-based retrieval.
 */
public class RawTextFileDocument extends TextFileDocument {

  /**
   * Creates a raw text document reader.
   *
   * @param file the text file to read
   * @param stem whether stemming should be applied to produced tokens
   */
  public RawTextFileDocument(File file, boolean stem) {
    super(file, stem);
  }

  /**
   * Advances {@code nextToken} to the next valid token.
   *
   * Processing steps:
   *
   *   1. Obtain the next candidate token from the text stream.
   *   2. Normalize to lowercase.
   *   3. Filter non-letter tokens using {@code allLetters}.
   *   4. Optionally apply stemming via {@code stemmer.stripAffixes} and revalidate.
   *   5. Emit the token without stopword filtering.
   * 
   *
   * If no further tokens exist or the token is invalid, {@code nextToken} is left as {@code null}.
   */
  @Override
  protected void prepareNextToken() {
    nextToken = getNextCandidateToken();
    if (nextToken == null) return;

    nextToken = nextToken.toLowerCase();

    if (!allLetters(nextToken)) {
      nextToken = null;
      return;
    }

    if (stem) {
      nextToken = stemmer.stripAffixes(nextToken);
      if (!allLetters(nextToken)) {
        nextToken = null;
      }
    }
  }
}
