syntax: numerator.y
	bison -H numerator.y

lexer: numerator.l
	lex numerator.l

numerator: syntax lexer helpers.c helpers.h tree.c tree.h
	gcc lex.yy.c numerator.tab.c helpers.c tree.c -onumerator -lm

clean:
	rm numerator lex.yy.c numerator.tab.c numerator.tab.h