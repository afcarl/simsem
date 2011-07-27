# Generate all SimString databases from our lexical resources.
#
# You might want to try `-j ${PROCESSES}` or get a coffee for this one.
#
# Author:	Pontus Stentorp	<pontus stenetorp se>
# Version:	2011-07-25

#TODO: Download the resource, also clean it

SIMSTRING_BIN:=simstring
DB_DIR:=../dbs
DATA_DIR:=simsem_lexical_resources

DB_SUFF:=db
# Additional "support" databases created by SimString
CDB_SUFF:=cdb

# Instantiate target variable
DB_TARGETS:=

# Convert path to db-path, create a target for the db and register it globally
define resource-target
RES_PATH:=$1
RES_FILE:=$${shell echo $${RES_PATH} | \
	sed -e 's|^[./]\+||g' -e 's|/|.|g' -e 's|${DATA_DIR}\.||g'}
DB_FILE:=$${addsuffix .${DB_SUFF}, $${RES_FILE}}
DB_PATH:=$${addprefix ${DB_DIR}/, $${DB_FILE}}

$${DB_PATH}: $${RES_PATH}
	cat $$< | ${SIMSTRING_BIN} --build --database $$@ --quiet

DB_TARGETS+=$${DB_PATH}
DB_TARGETS:=$${sort $${DB_TARGETS}}
endef

# Call the previously declared function for all tokenised resources
${foreach TOKENISED_RESOURCE, \
	${shell find ${DATA_DIR} -name '*.tokens'}, \
	${eval ${call resource-target, ${TOKENISED_RESOURCE}}} \
}

all: ${DB_TARGETS}

clean:
	find ${DB_DIR} -maxdepth 1 -name '*.${DB_SUFF}.*.${CDB_SUFF}' \
		-o -name '*.${DB_SUFF}' | xargs -r rm

.PHONY: all clean
.DEFAULT_GOAL:=all
