import java.io.File
import scala.collection.mutable.Map

/**
  * Generate Joern method-level CPG exports for a given source directory.
  *
  * For each method in the imported CPG, we write a dotCpg14 file into `outDir`,
  * using a de-duplicated method full name as the filename.
  */
def getJoern(srcDir: String, outDir: String) = {
  // Import the code for this directory into Joern
  importCode(inputPath = srcDir, projectName = "tmp")

  // Ensure the output directory exists
  val outDirFile = new File(outDir)
  if (!outDirFile.exists()) {
    outDirFile.mkdirs()
  }

  // Keep track of how many times we've seen each method fullName
  var nameCount: Map[String, Int] = Map()

  // For each method, emit a dotCpg14 file with a stable, unique name
  for (method <- cpg.method.l) {
    var fullName = method.fullName.split(":")(0)

    if (nameCount.contains(fullName)) {
      val cnt = nameCount(fullName)
      nameCount.update(fullName, cnt + 1)
      fullName = fullName + cnt
    } else {
      nameCount.update(fullName, 1)
    }

    // Write the method-level CPG, but don't let a single failure stop the script
    try {
      method.dotCpg14.l |> (outDir + "/" + fullName)
    } catch {
      case _: Throwable =>
        // Ignore individual method export errors
    }
  }
}

/**
  * Generic Joern generation driven by an external targets file.
  *
  * Python (e.g., gen_joern() in gen_graph.py) writes a file
  *   joern_targets.txt
  * in the same directory as this script, where each non-empty line is:
  *
  *   <srcDir>;<outDir>
  *
  * This main method simply iterates that list and invokes getJoern
  * for each (srcDir, outDir) pair.
  */
@main def genAllJoern() = {
  val cwd = new File(".").getAbsolutePath
  val targetsFile = new File("joern_targets.txt")

  if (!targetsFile.exists()) {
    println(s"Targets file not found: ${targetsFile.getAbsolutePath}")
    println("Expected a file 'joern_targets.txt' with lines of the form:")
    println("  <srcDir>;<outDir>")
    sys.exit(1)
  }

  val lines = scala.io.Source.fromFile(targetsFile).getLines().toList

  var totalProjects = 0

  for (raw <- lines) {
    val line = raw.trim
    if (line.nonEmpty && !line.startsWith("#")) {
      val parts = line.split(";", 2)
      if (parts.length == 2) {
        val srcDir = parts(0).trim
        val outDir = parts(1).trim

        val srcFile = new File(srcDir)
        if (!srcFile.exists() || !srcFile.isDirectory) {
          println(s"⚠️  Skipping non-existent or non-directory src: $srcDir")
        } else {
          val outFile = new File(outDir)
          outFile.mkdirs()
          try {
            println(s"Generating Joern CPG: src=$srcDir  out=$outDir")
            getJoern(srcDir, outDir)
            totalProjects += 1
          } catch {
            case ex: Throwable =>
              println(s"  ⚠️  Failed Joern for src=$srcDir: ${ex.getMessage}")
          }
        }
      }
    }
  }

  println("=" * 70)
  println(s"JOERN GENERATION COMPLETE - processed $totalProjects before/after projects")
  println("=" * 70)
}

