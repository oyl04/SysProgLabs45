%code top{
  #include <stdio.h> 
  #include <math.h>   
  #include "helpers.h"
  #include "tree.h"

  int yylex (void);
  void yyerror (char const *);
  void yyset_in (FILE *);

}

%locations
%define api.value.type { struct node* }
%token NUM     /* Double precision number. */
%token VAR UNARY_FUNC
%nterm exp

%precedence '='
%left '-' '+'
%left '*' '/' '%'
%precedence NEG /* negation--unary minus */
%right '^'      /* exponentiation */



%% /* The grammar follows. */
input:
  %empty
| input line
;

line:
  '\n'
| exp '\n'   { printf ("\033[1;35m%.10g\n\033[0m", $1->value); print_syntax_tree($1);}
| statement '\n'
| error '\n' { yyerrok;                }
;

statement:
  VAR '=' exp { add_variable($1->name, $3->value);    }

exp:
  NUM {
    $$ = create_node_value("num", $1->value);
    add_sub_node($$, $1);
  }
| VAR                
{
  variable_node* var = get_variable($1->name);
  if(var == NULL) {
    yyerror("Referencing undefined variable");
    YYERROR;
  }
  $$ = create_node("var");
  add_sub_node($$, $1);
  $$->value = var->value;
}
| UNARY_FUNC '(' exp ')' 
{
  $$ = create_node_list_3("exp", $1, $2, $3);
  add_sub_node($$, $4);
  if(!process_unary_function($1->name, $3->value, &$$->value)) {
    YYERROR;
  } 
}
| exp '+' exp        { $$ = create_node_list_3("exp", $1, $2, $3); $$->value = $1->value + $3->value;}
| exp '-' exp        { $$ = create_node_list_3("exp", $1, $2, $3); $$->value = $1->value - $3->value;}
| exp '*' exp        { $$ = create_node_list_3("exp", $1, $2, $3); $$->value = $1->value * $3->value;}
| exp '/' exp        { $$ = create_node_list_3("exp", $1, $2, $3); $$->value = $1->value / $3->value;}
| exp '%' exp        { $$ = create_node_list_3("exp", $1, $2, $3); $$->value = (int)$1->value % (int)$3->value;}
| '-' exp  %prec NEG { $$ = create_node_list_2("exp", $1, $2); $$->value = -$2->value;}
| exp '^' exp        { $$ = create_node_list_3("exp", $1, $2, $3); $$->value = pow ($1->value, $3->value);}
| '(' exp ')'        { $$ = create_node_list_3("exp", $1, $2, $3); $$->value = $2->value;}
;
/* End of grammar. */
%%

/* Called by yyparse on error. */
void yyerror (char const *s)
{
  fprintf (stderr, "%s\n", s);
}

int main(int argc, char const *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    FILE *input_file = fopen(argv[1], "r");
    if (!input_file) {
        fprintf(stderr, "Unable to open input file %s\n", argv[1]);
        return 1;
    }

    yyset_in(input_file);
    init_variable_table();
    int parse_result = yyparse();
    fclose(input_file);
    return parse_result;
}