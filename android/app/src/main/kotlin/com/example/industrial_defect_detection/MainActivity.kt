
package com.example.industrial_defect_detection

import android.os.Bundle
import io.flutter.FlutterInjector
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {

    companion object {
        private const val CHANNEL =
            "industrial_defect_detection/llama"

        // Updated 512-token model
        private const val MODEL_NAME =
            "llama3_2_1b_xnnpack_8da4w_512.pte"

        private const val TOKENIZER_NAME =
            "tokenizer.json"

        private const val MODEL_ASSET =
            "assets/models/llama/llama3_2_1b_xnnpack_8da4w_512.pte"

        private const val TOKENIZER_ASSET =
            "assets/models/llama/tokenizer.json"

        // Exact verified model size: 802247040 bytes
        private const val EXPECTED_MODEL_SIZE =
            802247040L

        private const val EXPECTED_TOKENIZER_SIZE =
            9085657L

        // Exported model context limit: 512 tokens
        private const val MAX_GENERATION_TOKENS = 100
    }

    private val llamaExecutor =
        Executors.newSingleThreadExecutor()

    private val copyExecutor =
        Executors.newSingleThreadExecutor()

    private val llamaTemperature = 0.7f

    @Volatile
    private var llamaModule: LlmModule? = null

    @Volatile
    private var isGenerating = false

    private val extractionLock = Any()

    override fun configureFlutterEngine(
        flutterEngine: FlutterEngine
    ) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(
            flutterEngine.dartExecutor.binaryMessenger,
            CHANNEL
        ).setMethodCallHandler { call, result ->

            when (call.method) {

                "getPlatformInfo" -> {
                    result.success(
                        mapOf(
                            "platform" to "Android",
                            "device" to android.os.Build.MODEL,
                            "androidVersion" to
                                android.os.Build.VERSION.RELEASE,
                            "architecture" to
                                android.os.Build.SUPPORTED_ABIS
                                    .firstOrNull()
                        )
                    )
                }

                "prepareLlamaFiles" -> {
                    prepareLlamaFiles(result)
                }

                "selectModelFiles" -> {
                    prepareLlamaFiles(result)
                }

                "loadLlamaModel" -> {
                    loadLlamaModel(result)
                }

                "generateLlamaText" -> {
                    val prompt =
                        call.argument<String>("prompt")

                    if (prompt.isNullOrBlank()) {
                        result.error(
                            "INVALID_PROMPT",
                            "Prompt cannot be empty.",
                            null
                        )
                    } else {
                        generateLlamaText(prompt, result)
                    }
                }

                else -> result.notImplemented()
            }
        }
    }

    // ---------------------------------------------------------
    // AUTOMATIC MODEL AND TOKENIZER EXTRACTION
    // ---------------------------------------------------------

    private fun prepareLlamaFiles(
        result: MethodChannel.Result
    ) {
        copyExecutor.execute {
            try {
                val files = ensureLlamaFiles()

                runOnUiThread {
                    result.success(
                        mapOf(
                            "success" to true,
                            "message" to "Llama Files Verified",
                            "modelPath" to
                                files.first.absolutePath,
                            "tokenizerPath" to
                                files.second.absolutePath,
                            "modelSize" to
                                files.first.length(),
                            "tokenizerSize" to
                                files.second.length()
                        )
                    )
                }

            } catch (e: Exception) {
                runOnUiThread {
                    result.error(
                        "LLAMA_ASSET_ERROR",
                        e.message
                            ?: "Unable to prepare Llama files.",
                        null
                    )
                }
            }
        }
    }

    private fun ensureLlamaFiles(): Pair<File, File> {

        synchronized(extractionLock) {

            val modelDirectory =
                File(cacheDir, "llama_model")

            if (!modelDirectory.exists()) {
                check(modelDirectory.mkdirs()) {
                    "Unable to create Llama model directory."
                }
            }

            val modelFile =
                File(modelDirectory, MODEL_NAME)

            val tokenizerFile =
                File(modelDirectory, TOKENIZER_NAME)

            if (
                !modelFile.exists() ||
                modelFile.length() != EXPECTED_MODEL_SIZE
            ) {
                copyFlutterAsset(
                    MODEL_ASSET,
                    modelFile,
                    EXPECTED_MODEL_SIZE
                )
            }

            if (
                !tokenizerFile.exists() ||
                tokenizerFile.length() != EXPECTED_TOKENIZER_SIZE
            ) {
                copyFlutterAsset(
                    TOKENIZER_ASSET,
                    tokenizerFile,
                    EXPECTED_TOKENIZER_SIZE
                )
            }

            check(
                modelFile.exists() &&
                    modelFile.length() == EXPECTED_MODEL_SIZE
            ) {
                "Llama model verification failed."
            }

            check(
                tokenizerFile.exists() &&
                    tokenizerFile.length() == EXPECTED_TOKENIZER_SIZE
            ) {
                "Tokenizer verification failed."
            }

            return Pair(modelFile, tokenizerFile)
        }
    }

    private fun copyFlutterAsset(
        assetPath: String,
        destination: File,
        expectedSize: Long
    ) {

        val assetKey =
            FlutterInjector.instance()
                .flutterLoader()
                .getLookupKeyForAsset(assetPath)

        val temporaryFile =
            File(destination.parentFile, destination.name + ".tmp")

        try {
            applicationContext.assets
                .open(assetKey)
                .use { input ->

                    temporaryFile.outputStream()
                        .buffered()
                        .use { output ->
                            input.copyTo(
                                output,
                                1024 * 1024
                            )
                        }
                }

            check(temporaryFile.length() == expectedSize) {
                "Incorrect file size for ${destination.name}. " +
                    "Expected $expectedSize bytes, " +
                    "received ${temporaryFile.length()} bytes."
            }

            if (destination.exists()) {
                check(destination.delete()) {
                    "Unable to replace ${destination.name}."
                }
            }

            check(temporaryFile.renameTo(destination)) {
                "Unable to save ${destination.name}."
            }

        } finally {
            if (temporaryFile.exists()) {
                temporaryFile.delete()
            }
        }
    }

    // ---------------------------------------------------------
    // LOAD LLAMA MODEL
    // ---------------------------------------------------------

    private fun loadLlamaModel(
        result: MethodChannel.Result
    ) {

        llamaExecutor.execute {

            try {
                if (llamaModule != null) {
                    runOnUiThread {
                        result.success(
                            mapOf(
                                "success" to true,
                                "message" to
                                    "Llama model is already loaded."
                            )
                        )
                    }
                    return@execute
                }

                val files = ensureLlamaFiles()

                val modelFile = files.first
                val tokenizerFile = files.second

                val module = LlmModule(
                    LlmModule.MODEL_TYPE_TEXT,
                    modelFile.absolutePath,
                    tokenizerFile.absolutePath,
                    llamaTemperature
                )

                try {
                    module.load()
                    llamaModule = module
                } catch (e: Exception) {
                    module.close()
                    throw e
                }

                runOnUiThread {
                    result.success(
                        mapOf(
                            "success" to true,
                            "message" to
                                "Llama model loaded successfully."
                        )
                    )
                }

            } catch (e: Exception) {

                runOnUiThread {
                    result.error(
                        "LLAMA_LOAD_ERROR",
                        e.message
                            ?: "Unable to load Llama model.",
                        null
                    )
                }
            }
        }
    }

    // ---------------------------------------------------------
    // GENERATE TEXT USING LLAMA
    // ---------------------------------------------------------

    private fun generateLlamaText(
        prompt: String,
        result: MethodChannel.Result
    ) {

        llamaExecutor.execute {

            try {
                val module = llamaModule
                    ?: throw IllegalStateException(
                        "Load the Llama model before generating text."
                    )

                isGenerating = true

                val generatedText = StringBuilder()
                var callbackError: String? = null

                val callback = object : LlmCallback {

                    override fun onResult(resultText: String) {
                        generatedText.append(resultText)
                    }

                    override fun onStats(stats: String) {
                        android.util.Log.d(
                            "LlamaStats",
                            stats
                        )
                    }

                    override fun onError(
                        errorCode: Int,
                        message: String
                    ) {
                        callbackError =
                            "Error $errorCode: $message"
                    }
                }

                try {
                    module.generate(
                        prompt,
                        512, // Total sequence length; model context limit is 512.
                        callback,
                        false,
                        llamaTemperature,
                        0,
                        0
                    )
                } finally {
                    isGenerating = false
                }

                if (callbackError != null) {
                    throw RuntimeException(callbackError)
                }

                val response = generatedText.toString()
                    .replace("<|eot_id|>", "")
                    .replace("<|end_of_text|>", "")
                    .trim()

                if (response.isEmpty()) {
                    throw RuntimeException(
                        "Llama returned an empty response."
                    )
                }

                runOnUiThread {
                    result.success(
                        mapOf(
                            "success" to true,
                            "response" to response
                        )
                    )
                }

            } catch (e: Exception) {

                isGenerating = false

                runOnUiThread {
                    result.error(
                        "LLAMA_GENERATION_ERROR",
                        e.message
                            ?: "Text generation failed.",
                        null
                    )
                }
            }
        }
    }

    // ---------------------------------------------------------
    // RELEASE LLAMA RESOURCES
    // ---------------------------------------------------------

    override fun onDestroy() {

        if (isGenerating) {
            llamaModule?.stop()
        }

        if (!llamaExecutor.isShutdown) {

            llamaExecutor.execute {
                try {
                    llamaModule?.close()
                } catch (e: Exception) {
                    android.util.Log.e(
                        "Llama",
                        "Model cleanup failed",
                        e
                    )
                } finally {
                    llamaModule = null
                }
            }
        }

        llamaExecutor.shutdown()
        copyExecutor.shutdown()

        super.onDestroy()
    }
}
