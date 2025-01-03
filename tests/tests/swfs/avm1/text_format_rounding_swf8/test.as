function test(v) {
    var tf = new TextFormat(v, v, v, v, v, v, v, v, v, v, v, v, v);
    trace("value " + v);
    trace("  font = " + tf.font);
    trace("  size = " + tf.size);
    trace("  color = " + tf.color);
    trace("  bold = " + tf.bold);
    trace("  italic = " + tf.italic);
    trace("  underline = " + tf.underline);
    trace("  url = " + tf.url);
    trace("  target = " + tf.target);
    trace("  align = " + tf.align);
    trace("  leftMargin = " + tf.leftMargin);
    trace("  rightMargin = " + tf.rightMargin);
    trace("  indent = " + tf.indent);
    trace("  leading = " + tf.leading);
    var tf2 = new TextFormat("font", 3, 14, true, false, true, "http://example.com", "_blank", "right", 4, 6, 9, 1);
    tf2.font = v;
    tf2.size = v;
    tf2.color = v;
    tf2.bold = v;
    tf2.italic = v;
    tf2.underline = v;
    tf2.url = v;
    tf2.target = v;
    tf2.align = v;
    tf2.leftMargin = v;
    tf2.rightMargin = v;
    tf2.indent = v;
    tf2.leading = v;
    tf2.blockIndent = v;
    trace("  font 2 = " + tf2.font);
    trace("  size 2 = " + tf2.size);
    trace("  color 2 = " + tf2.color);
    trace("  bold 2 = " + tf2.bold);
    trace("  italic 2 = " + tf2.italic);
    trace("  underline 2 = " + tf2.underline);
    trace("  url 2 = " + tf2.url);
    trace("  target 2 = " + tf2.target);
    trace("  align 2 = " + tf2.align);
    trace("  leftMargin 2 = " + tf2.leftMargin);
    trace("  rightMargin 2 = " + tf2.rightMargin);
    trace("  indent 2 = " + tf2.indent);
    trace("  leading 2 = " + tf2.leading);
    trace("  blockIndent 2 = " + tf2.blockIndent);
}

test(-11);
test(-10.9);
test(-10.5);
test(-10.1);
test(-10);
test(-9.9);
test(-9.5);
test(-9.1);
test(-1);
test(0)
test(0.1);
test(0.5);
test(0.9);
test(1.1);
test(1.5);
test(1.9);
test(19.1);
test(19.5);
test(19.9);
test(null);
test(undefined);
test("1");
test("1.1");
test("1.5");
test("1.9");
test("2.1");
test("2.5");
test("2.9");
test(true);
test(false);
