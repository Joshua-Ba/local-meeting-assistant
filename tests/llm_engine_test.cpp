#include "llm_engine.h"

#include "test_helpers.h"

#include <gtest/gtest.h>

TEST(LlmEngineTest, MissingModelLeavesEngineUnloaded) {
    Config config = make_test_config();
    config.model_path = "__definitely_missing_model__.gguf";

    LlmEngine engine(config);

    EXPECT_FALSE(engine.is_loaded());
    EXPECT_EQ(engine.generate("input", 8, "prompt"), std::nullopt);
}
