#include "utils.h"

#include <gtest/gtest.h>

#include <regex>
#include <string>

TEST(UtilsTest, GetFilepathBuildsExpectedOutputPath) {
    EXPECT_EQ(get_filepath("output", "summary", "2026_04_21_10_30"),
              "output/summary_2026_04_21_10_30.txt");
}

TEST(UtilsTest, GenerateSessionIdUsesStableTimestampFormat) {
    const std::string session_id = generate_session_id();
    const std::regex pattern(R"(^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}$)");

    EXPECT_TRUE(std::regex_match(session_id, pattern));
}
