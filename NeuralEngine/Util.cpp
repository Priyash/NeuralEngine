#include"Util.h"

Util* Util::util = 0;


Util::Util()
{
}

//RETURNS VALUE FOR A GIVEN OBJECT NAME
JSON_VALUE Util::getValue(CONFIG con, string key)
{
	const Value& Shape_Obj = doc[toStr(con).c_str()];
	JSON_VALUE jv;
	if (Shape_Obj.IsString())
	{
		jv.json_str_value = Shape_Obj[key.c_str()].GetString();
	}
	else if (Shape_Obj.IsInt())
	{
		jv.json_int_value = Shape_Obj[key.c_str()].GetInt();
	}

	return jv;
}


//RETURNS LIST OF MEMBERS FOR A GIVEN OBJECT NAME
vector<JSON_VALUE> Util::getValues(CONFIG con)
{
	vector<JSON_VALUE>values_list;
	for (Value::ConstMemberIterator itr = doc[toStr(con).c_str()].MemberBegin(); itr != doc[toStr(con).c_str()].MemberEnd(); ++itr)
	{
		JSON_VALUE jv;
		if (itr->value.IsString())
		{
			jv.json_str_value = itr->value.GetString();
		}
		else if (itr->value.IsInt())
		{
			jv.json_int_value = itr->value.GetInt();
		}
		values_list.push_back(jv);
	}

	return values_list;
}

//CHECK THIS https://stackoverflow.com/questions/30896857/iterate-and-retrieve-nested-object-in-json-using-rapidjson 
//FOR THIS PROB

vector<JSON_VALUE> Util::getValues(Value::ConstMemberIterator obj)
{
	vector<JSON_VALUE>values_list;
	for (Value::ConstMemberIterator itr = obj->value.MemberBegin(); itr != obj->value.MemberEnd(); ++itr)
	{
		JSON_VALUE jv;
		if (itr->value.IsString())
		{
			jv.json_str_value = itr->value.GetString();
		}
		else if (itr->value.IsInt())
		{
			jv.json_int_value = itr->value.GetInt();
		}
		values_list.push_back(jv);
	}

	return values_list;
}


vector<Value::ConstMemberIterator>Util::getObjects(CONFIG con)
{
	vector<Value::ConstMemberIterator>object_list;
	const Value& object = doc[toStr(con).c_str()];
	for (Value::ConstMemberIterator itr = object.MemberBegin(); itr != object.MemberEnd(); ++itr)
	{
		if (itr->value.IsObject())
		{
			cout << itr->name.GetString() << endl;
			object_list.push_back(itr);
		}
	}
	return object_list;
}



string Util::toStr(CONFIG con)
{
	string s = "";
	switch (con)
	{
	case CONVLAYER1:
		s = "ConvLayer1";
		break;
	case INPUT_SHAPE:
		s = "Input_Shape";
		break;
	default:
		break;
	}

	return s;
}


Util* Util::getInstance()
{
	if (!util)
	{
		util = new Util();
		return util;
	}
	else
	{
		return util;
	}
}


void Util::read_Json()
{
	config_file = "config.json";
	ifstream json_reader(config_file);
	IStreamWrapper isw(json_reader);
	doc.ParseStream(isw);
}