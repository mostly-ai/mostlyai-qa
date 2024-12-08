# Copyright 2024 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mostlyai.qa._accuracy import format_display_prefixes, trim_label


def test_format_display_prefixes():
    labels = ["ctx::A", "tgt::B", "nxt::C", "D"]
    labels = format_display_prefixes(*labels)
    assert labels == ["context:A", "B", "C", "D"]


def test_trim_label():
    assert trim_label("a" * 40, max_length=10) == "a" * 4 + "..." + "a" * 3
    assert trim_label("a" * 40, max_length=11) == "a" * 4 + "..." + "a" * 4
    assert trim_label("a" * 40, max_length=12) == "a" * 5 + "..." + "a" * 4
    assert trim_label("a" * 6, max_length=5, reserved_labels={"a...a"}) == "a...a0"
    assert trim_label("a" * 6, max_length=5, reserved_labels={"a...a", "a...a0"}) == "a...a1"
